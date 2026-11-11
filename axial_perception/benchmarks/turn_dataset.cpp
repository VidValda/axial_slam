#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "bench.hpp"
#include "axial_perception/masks.hpp"
#include "axial_perception/occlusion.hpp"
#include "axial_perception/segments.hpp"

using namespace axial_perception;
using namespace axial_perception::bench;

namespace
{

constexpr int kArmPoints = 20;
constexpr double kCorner = 1.0;
constexpr double kInf = std::numeric_limits<double>::infinity();

struct Joint
{
  int i;
  int j;
  Eigen::Vector3d at;
};

std::vector<Joint> joints(const Sections & s)
{
  std::vector<Joint> out;
  for (size_t i = 0; i < s.size(); ++i) {
    for (size_t j = i + 1; j < s.size(); ++j) {
      if (s.run[i] != s.run[j]) {
        continue;
      }
      const Eigen::Vector3d ends_i[2] = {s.a[i], s.b[i]};
      const Eigen::Vector3d ends_j[2] = {s.a[j], s.b[j]};
      double best = kInf;
      int arg = 0;
      for (int p = 0; p < 2; ++p) {
        for (int q = 0; q < 2; ++q) {
          const double d = (ends_i[p] - ends_j[q]).norm();
          if (d < best) {
            best = d;
            arg = p;
          }
        }
      }
      if (best < 0.05) {
        out.push_back({static_cast<int>(i), static_cast<int>(j), ends_i[arg]});
      }
    }
  }
  return out;
}

std::vector<bool> elbow_zone(
  const Cloud & p, const std::vector<int> & g, const Sections & s, const std::vector<Joint> & turns)
{
  std::vector<bool> near(g.size(), false);
  for (const auto & t : turns) {
    const double r = std::max(s.r[static_cast<size_t>(t.i)], s.r[static_cast<size_t>(t.j)]) + 0.05;
    for (size_t k = 0; k < g.size(); ++k) {
      if ((g[k] == t.i || g[k] == t.j) && (p.col(static_cast<Eigen::Index>(k)) - t.at).norm() < r) {
        near[k] = true;
      }
    }
  }
  return near;
}

std::vector<int> gt_labels(
  const Cloud & world, const std::vector<int> & cls, const std::vector<int> & seg, const Sections & s,
  double tol = 0.1)
{
  const std::vector<int> g = section_ids(world, cls, s, tol);
  std::vector<int> out(g.size(), -1);
  std::map<int, Indices> masks;
  for (size_t i = 0; i < seg.size(); ++i) {
    masks[seg[i]].push_back(static_cast<int>(i));
  }
  for (const auto & [m, sel] : masks) {
    std::vector<int> runs;
    for (int i : sel) {
      if (g[static_cast<size_t>(i)] >= 0) {
        runs.push_back(s.run[static_cast<size_t>(g[static_cast<size_t>(i)])]);
      }
    }
    if (runs.empty()) {
      continue;
    }
    const int main = argmax_count(runs);
    Indices own;
    for (size_t k = 0; k < s.size(); ++k) {
      if (s.run[k] == main) {
        own.push_back(static_cast<int>(k));
      }
    }
    std::vector<int> sub_cls;
    for (int i : sel) {
      sub_cls.push_back(cls[static_cast<size_t>(i)]);
    }
    const std::vector<int> mine = section_ids(select(world, sel), sub_cls, s, tol, own);
    for (size_t k = 0; k < sel.size(); ++k) {
      const size_t i = static_cast<size_t>(sel[k]);
      out[i] = mine[k] >= 0 ? mine[k] : g[i] >= 0 ? -2 : -1;
    }
  }
  return out;
}

std::vector<uint8_t> thin(const std::vector<uint8_t> & mask, int h, int w)
{
  const int W = w + 2;
  std::vector<uint8_t> img(static_cast<size_t>((h + 2) * W), 0);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      img[static_cast<size_t>((y + 1) * W + x + 1)] = mask[static_cast<size_t>(y * w + x)];
    }
  }
  auto px = [&](int y, int x) {return static_cast<int>(img[static_cast<size_t>(y * W + x)]);};
  std::vector<size_t> drop;
  while (true) {
    bool changed = false;
    for (int step = 0; step < 2; ++step) {
      drop.clear();
      for (int y = 1; y <= h; ++y) {
        for (int x = 1; x <= w; ++x) {
          if (!px(y, x)) {
            continue;
          }
          const int n[8] = {px(y - 1, x), px(y - 1, x + 1), px(y, x + 1), px(y + 1, x + 1), px(y + 1, x),
            px(y + 1, x - 1), px(y, x - 1), px(y - 1, x - 1)};
          int b = 0;
          int a = 0;
          for (int k = 0; k < 8; ++k) {
            b += n[k];
            a += n[k] == 0 && n[(k + 1) % 8] == 1;
          }
          const bool side = step == 0 ?
            n[0] * n[2] * n[4] == 0 && n[2] * n[4] * n[6] == 0 :
            n[0] * n[2] * n[6] == 0 && n[0] * n[4] * n[6] == 0;
          if (b >= 2 && b <= 6 && a == 1 && side) {
            drop.push_back(static_cast<size_t>(y * W + x));
          }
        }
      }
      for (size_t i : drop) {
        img[i] = 0;
      }
      changed = changed || !drop.empty();
    }
    if (!changed) {
      break;
    }
  }
  std::vector<uint8_t> out(static_cast<size_t>(h * w));
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      out[static_cast<size_t>(y * w + x)] = img[static_cast<size_t>((y + 1) * W + x + 1)];
    }
  }
  return out;
}

struct Piece
{
  Eigen::Vector2d a;
  Eigen::Vector2d b;
  double r;
};

std::vector<Piece> centerline_pieces(
  const std::vector<uint8_t> & mask, int h, int w, double eps_ratio = 0.5, int min_pixels = 5)
{
  cv::Mat padded = cv::Mat::zeros(h + 2, w + 2, CV_8U);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      padded.at<uint8_t>(y + 1, x + 1) = mask[static_cast<size_t>(y * w + x)];
    }
  }
  cv::Mat radius;
  cv::distanceTransform(padded, radius, cv::DIST_L2, 3);
  const std::vector<uint8_t> sk = thin(mask, h, w);
  Cloud pts(3, 0);
  std::vector<Eigen::Vector3d> v;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      if (sk[static_cast<size_t>(y * w + x)]) {
        v.emplace_back(x, y, 0.0);
      }
    }
  }
  std::vector<Piece> pieces;
  if (static_cast<int>(v.size()) < min_pixels) {
    return pieces;
  }
  pts.resize(3, static_cast<Eigen::Index>(v.size()));
  for (size_t i = 0; i < v.size(); ++i) {
    pts.col(static_cast<Eigen::Index>(i)) = v[i];
  }
  const Mst tree = mst(pts);
  std::vector<int> comp(v.size());
  for (size_t i = 0; i < comp.size(); ++i) {
    comp[i] = static_cast<int>(i);
  }
  for (size_t m = 1; m < tree.order.size(); ++m) {
    const size_t k = static_cast<size_t>(tree.order[m]);
    if (tree.w[k] < 1.5) {
      comp[k] = comp[static_cast<size_t>(tree.parent[k])];
    }
  }
  std::map<int, Indices> by_comp;
  for (size_t i = 0; i < comp.size(); ++i) {
    by_comp[comp[i]].push_back(static_cast<int>(i));
  }
  for (const auto & [c, nodes] : by_comp) {
    if (static_cast<int>(nodes.size()) < min_pixels) {
      continue;
    }
    const Indices path = longest_path(nodes, tree.parent, pts);
    std::vector<double> rs;
    std::vector<cv::Point2f> line;
    for (int k : path) {
      const int x = static_cast<int>(pts(0, k));
      const int y = static_cast<int>(pts(1, k));
      rs.push_back(radius.at<float>(y + 1, x + 1));
      line.emplace_back(static_cast<float>(x), static_cast<float>(y));
    }
    const double r = median(rs);
    std::vector<cv::Point2f> poly;
    cv::approxPolyDP(line, poly, std::max(2.0, eps_ratio * r), false);
    for (size_t k = 0; k + 1 < poly.size(); ++k) {
      pieces.push_back({{poly[k].x, poly[k].y}, {poly[k + 1].x, poly[k + 1].y}, r});
    }
  }
  return pieces;
}

Labels split_2d(const FramePoints & fp, const RawFrame & f, const Params & p)
{
  const InstanceImage img = decode_gz_instance(f.labels, f.height, f.width, 3 * static_cast<size_t>(f.width));
  const Projection proj = project(fp.xyz, f.K, img.width, img.height, 0.0, kInf);
  if (proj.size() != static_cast<size_t>(fp.xyz.cols())) {
    throw std::runtime_error("mask points out of the image");
  }
  Labels line(fp.seg.size(), 0);
  std::map<int, Indices> masks;
  for (size_t i = 0; i < fp.seg.size(); ++i) {
    masks[fp.seg[i]].push_back(static_cast<int>(i));
  }
  for (const auto & [s, sel] : masks) {
    const size_t first = static_cast<size_t>(sel.front());
    const int32_t id = img.mask_at(proj.u[first], proj.v[first]);
    int x0 = img.width, y0 = img.height, x1 = -1, y1 = -1;
    for (int y = 0; y < img.height; ++y) {
      for (int x = 0; x < img.width; ++x) {
        if (img.mask_at(x, y) == id) {
          x0 = std::min(x0, x);
          x1 = std::max(x1, x);
          y0 = std::min(y0, y);
          y1 = std::max(y1, y);
        }
      }
    }
    const int cw = x1 - x0 + 1;
    const int ch = y1 - y0 + 1;
    std::vector<uint8_t> crop(static_cast<size_t>(cw * ch), 0);
    for (int y = y0; y <= y1; ++y) {
      for (int x = x0; x <= x1; ++x) {
        crop[static_cast<size_t>((y - y0) * cw + x - x0)] = img.mask_at(x, y) == id;
      }
    }
    const auto pieces = centerline_pieces(crop, ch, cw);
    if (pieces.empty()) {
      continue;
    }
    Labels lab(sel.size());
    std::vector<Eigen::Vector2d> q(sel.size());
    for (size_t k = 0; k < sel.size(); ++k) {
      const size_t i = static_cast<size_t>(sel[k]);
      q[k] = Eigen::Vector2d(proj.u[i] - x0, proj.v[i] - y0);
      double best = kInf;
      for (size_t m = 0; m < pieces.size(); ++m) {
        const Eigen::Vector2d ab = pieces[m].b - pieces[m].a;
        const double t = std::clamp((q[k] - pieces[m].a).dot(ab) / std::max(ab.dot(ab), 1e-9), 0.0, 1.0);
        const double d = (q[k] - pieces[m].a - t * ab).norm();
        if (d < best) {
          best = d;
          lab[k] = static_cast<int>(m) + 1;
        }
      }
    }
    for (size_t m = 0; m < pieces.size(); ++m) {
      const double r = kCorner * pieces[m].r;
      if ((pieces[m].b - pieces[m].a).norm() < r) {
        std::replace(lab.begin(), lab.end(), static_cast<int>(m) + 1, 0);
      }
      if (m > 0 && pieces[m - 1].b == pieces[m].a) {
        for (size_t k = 0; k < sel.size(); ++k) {
          if ((q[k] - pieces[m].a).norm() < r) {
            lab[k] = 0;
          }
        }
      }
    }
    const int c = fp.cls[first];
    const Labels fin = finish(select(fp.xyz, sel), lab, p.tols.at(c), p.split_min_points, p.gaps.at(c));
    const int offset = std::max(0, *std::max_element(line.begin(), line.end()));
    for (size_t k = 0; k < sel.size(); ++k) {
      if (fin[k] > 0) {
        line[static_cast<size_t>(sel[k])] = fin[k] + offset;
      }
    }
  }
  return p.min_length > 0 ? drop_short(fp.xyz, line, p.min_length) : line;
}

std::vector<RawFrame> load(const std::string & path)
{
  DatasetReader in(path, Kind::Raw);
  std::vector<RawFrame> out;
  RawFrame f;
  while (in.next(f)) {
    out.push_back(f);
  }
  return out;
}

void build(const std::string & raw, const std::string & out, const Sections & s, const Params & p)
{
  constexpr size_t per_key = 4;
  constexpr double apart = 1.0;
  const auto turns = joints(s);
  DatasetReader in(raw, Kind::Raw);
  if (const auto dir = std::filesystem::absolute(out).parent_path(); !dir.empty()) {
    std::filesystem::create_directories(dir);
  }
  DatasetWriter writer(out, Kind::Raw);
  std::map<int, std::vector<Eigen::Vector3d>> views;
  size_t n = 0;
  RawFrame f;
  while (in.next(f)) {
    ++n;
    const FramePoints fp = frame_points(f, p, "none");
    const Cloud world = f.pose.apply(fp.xyz);
    const auto g = gt_labels(world, fp.cls, fp.seg, s);
    const auto elbow = elbow_zone(world, g, s, turns);
    std::map<int, std::map<int, std::vector<double>>> along;
    for (size_t i = 0; i < g.size(); ++i) {
      if (g[i] >= 0 && !elbow[i]) {
        along[fp.seg[i]][g[i]].push_back(
          world.col(static_cast<Eigen::Index>(i)).dot(s.axis(static_cast<size_t>(g[i]))));
      }
    }
    std::vector<int> keys;
    for (const auto & [seg, by_section] : along) {
      std::set<int> arms;
      for (const auto & [k, t] : by_section) {
        if (static_cast<int>(t.size()) >= kArmPoints &&
          *std::max_element(t.begin(), t.end()) - *std::min_element(t.begin(), t.end()) >= 0.15)
        {
          arms.insert(k);
        }
      }
      for (size_t j = 0; j < turns.size(); ++j) {
        if (arms.count(turns[j].i) && arms.count(turns[j].j)) {
          keys.push_back(static_cast<int>(j));
        }
      }
    }
    std::vector<int> fresh;
    const std::vector<Eigen::Vector3d> none;
    for (int k : keys) {
      const auto it = views.find(k);
      const auto & seen = it == views.end() ? none : it->second;
      if (seen.size() < per_key &&
        std::all_of(seen.begin(), seen.end(), [&](const Eigen::Vector3d & c) {return (f.pose.t - c).norm() >= apart;}))
      {
        fresh.push_back(k);
      }
    }
    if (!fresh.empty()) {
      for (int k : fresh) {
        views[k].push_back(f.pose.t);
      }
      writer.add(f);
    }
  }
  size_t total_views = 0;
  for (const auto & [k, v] : views) {
    total_views += v.size();
  }
  std::printf(
    "%s: %zu of %zu frames, %zu of %zu joints seen (%zu views), %.1f MB\n", out.c_str(), writer.frames(), n,
    views.size(), turns.size(), total_views, static_cast<double>(std::filesystem::file_size(out)) / 1e6);
}

void score_occlusion(const std::vector<RawFrame> & frames, const Sections & s, const Params & p, const std::string & o)
{
  double own = 0, other = 0, off = 0, base_own = 0;
  std::vector<double> times;
  for (const auto & f : frames) {
    const auto start = std::chrono::steady_clock::now();
    const FramePoints fp = frame_points(f, p, o);
    times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    for (int g : gt_labels(f.pose.apply(fp.xyz), fp.cls, fp.seg, s)) {
      own += g >= 0;
      other += g == -2;
      off += g == -1;
    }
    if (o != "none") {
      const FramePoints base = frame_points(f, p, "none");
      for (int g : gt_labels(f.pose.apply(base.xyz), base.cls, base.seg, s)) {
        base_own += g >= 0;
      }
    }
  }
  const double total = std::max(own + other + off, 1.0);
  const double recall = base_own > 0 ? 100 * own / base_own : 100.0;
  std::printf(
    "%-10s %8.1f %10.1f %12.1f %7.1f %9.1f\n", o.c_str(), 100 * own / total, 100 * other / total, 100 * off / total,
    recall, 1e3 * mean(times));
}

void score_split(
  const std::vector<RawFrame> & frames, const Sections & s, const Params & p, const std::string & o,
  const std::string & method)
{
  const auto turns = joints(s);
  std::map<std::pair<int, int>, Eigen::Vector3d> at;
  for (const auto & t : turns) {
    at[{t.i, t.j}] = t.at;
    at[{t.j, t.i}] = t.at;
  }
  double ok = 0, merged = 0, missed = 0, total = 0, wrong = 0, scored = 0, covered = 0, arm_points = 0;
  std::vector<double> depth, angles, times;
  for (const auto & f : frames) {
    const FramePoints fp = frame_points(f, p, o);
    const Cloud world = f.pose.apply(fp.xyz);
    const auto g = gt_labels(world, fp.cls, fp.seg, s);
    const auto start = std::chrono::steady_clock::now();
    const Labels line = method == "mask2d" ? split_2d(fp, f, p) :
      split_clusters(fp.xyz, fp.cls, fp.seg, method, p.tols, p.split_min_points, p.gaps, p.min_length);
    times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    const auto elbow = elbow_zone(world, g, s, turns);
    std::vector<bool> use(g.size());
    for (size_t i = 0; i < g.size(); ++i) {
      use[i] = g[i] >= 0 && !elbow[i];
    }
    std::map<int, int> main;
    for (const auto & [k, sel] : groups(line)) {
      std::vector<int> on;
      for (int i : sel) {
        if (use[static_cast<size_t>(i)]) {
          on.push_back(g[static_cast<size_t>(i)]);
        }
      }
      if (on.empty()) {
        continue;
      }
      const int m = argmax_count(on);
      main[k] = m;
      double reach = 0.0;
      for (int i : sel) {
        const size_t u = static_cast<size_t>(i);
        if (!use[u]) {
          continue;
        }
        ++scored;
        if (g[u] != m) {
          ++wrong;
          const auto j = at.find({m, g[u]});
          reach = std::max(
            reach, j == at.end() ? kInf :
            std::abs((world.col(i) - j->second).dot(s.axis(static_cast<size_t>(g[u])))));
        }
      }
      depth.push_back(reach);
      angles.push_back(
        std::acos(std::min(std::abs(fit_line(select(world, sel)).d.dot(s.axis(static_cast<size_t>(m)))), 1.0)) *
        180 / M_PI);
    }
    for (size_t i = 0; i < g.size(); ++i) {
      arm_points += use[i];
      covered += use[i] && line[i] > 0;
    }
    std::map<int, Indices> masks;
    for (size_t i = 0; i < fp.seg.size(); ++i) {
      if (use[i]) {
        masks[fp.seg[i]].push_back(static_cast<int>(i));
      }
    }
    for (const auto & [seg, sel] : masks) {
      for (const auto & t : turns) {
        std::vector<int> li, lj;
        for (int i : sel) {
          const size_t u = static_cast<size_t>(i);
          if (g[u] == t.i) {
            li.push_back(line[u]);
          } else if (g[u] == t.j) {
            lj.push_back(line[u]);
          }
        }
        if (static_cast<int>(li.size()) < kArmPoints || static_cast<int>(lj.size()) < kArmPoints) {
          continue;
        }
        ++total;
        const int a = argmax_count(li);
        const int b = argmax_count(lj);
        if (!a || !b) {
          ++missed;
        } else if (a == b) {
          ++merged;
        } else {
          bool clean = true;
          for (int k : {a, b}) {
            double hit = 0, n = 0;
            for (size_t i = 0; i < g.size(); ++i) {
              if (line[i] == k && use[i]) {
                ++n;
                hit += g[i] == main.at(k);
              }
            }
            clean = clean && hit / n >= 0.98;
          }
          ok += clean;
        }
      }
    }
  }
  std::vector<double> finite;
  size_t invading = 0;
  for (double d : depth) {
    invading += d > 0.1;
    if (std::isfinite(d)) {
      finite.push_back(d);
    }
  }
  const double split = std::max(total - missed, 1.0);
  std::printf(
    "%-10s %-9s %6d %7.1f %6.1f %6.1f %7.1f %6.1f %9.1f %8.2f %8.1f %9.1f %5.1f\n", o.c_str(), method.c_str(),
    static_cast<int>(total), 100 * missed / std::max(total, 1.0), 100 * ok / split,
    100 * (split - ok - merged) / split, 100 * merged / split, 100 * wrong / std::max(scored, 1.0),
    depth.empty() ? 0.0 : 100.0 * static_cast<double>(invading) / static_cast<double>(depth.size()),
    finite.empty() ? 0.0 : percentile(finite, 95), angles.empty() ? 0.0 : percentile(angles, 95),
    100 * covered / std::max(arm_points, 1.0), 1e3 * mean(times));
}

void run(const Args & args)
{
  const World world = load_world(args.required("--gt"));
  const Params p = load_params(args.required("--config"));
  const std::string data = args.required("--data");
  if (args.has("--build")) {
    build(args.str("--build"), data, world.sections, p);
    return;
  }
  const auto occlusions = args.list("--occlusion", {"none", "zbuffer"});
  const auto methods = args.list("--methods", {"none", "ransac"});
  for (const auto & o : occlusions) {
    check_occlusion_method(o);
  }
  for (const auto & m : methods) {
    if (m != "mask2d") {
      split_method(m);
    }
  }
  const auto frames = load(data);
  std::string tols, gaps;
  for (const auto & [c, v] : p.tols) {
    tols += (tols.empty() ? "" : ", ") + std::to_string(c) + ": " + std::to_string(v);
  }
  for (const auto & [c, v] : p.gaps) {
    gaps += (gaps.empty() ? "" : ", ") + std::to_string(c) + ": " + std::to_string(v);
  }
  std::printf(
    "%zu frames; occlusion window %d px, tol %g m; split tol {%s}, gap {%s}, min_points %d, min_length %g\n",
    frames.size(), p.projector.occlusion_window, p.projector.occlusion_tol, tols.c_str(), gaps.c_str(),
    p.split_min_points, p.min_length);
  std::printf("occlusion   own run  other run  off surface  recall  ms/frame\n");
  for (const auto & o : occlusions) {
    score_occlusion(frames, world.sections, p, o);
  }
  std::printf(
    "occlusion  method    joints  missed  clean  dirty  merged  wrong  invading  depth95  angle95  coverage  ms\n");
  for (const auto & o : occlusions) {
    for (const auto & m : methods) {
      score_split(frames, world.sections, p, o, m);
    }
  }
  std::printf(
    "own run / other run / off surface: kept mask points within 0.1 m of their mask's run, of another run, of "
    "neither;\nrecall: own-run points kept, against no filter. Turns: missed = an arm without a line; of the "
    "others,\nclean = each arm its own line, 98%% or more on it; dirty = less pure; merged = both arms on one "
    "line;\nwrong: line points on another section; invading / depth95: lines reaching more than 0.1 m into "
    "another section,\n95th percentile of that reach in m; angle95: 95th percentile of line tilt in deg\n");
}

}  // namespace

int main(int argc, char ** argv)
{
  std::vector<std::string> argv_all(argv, argv + argc);
  try {
    run(Args(argv_all, {{"--build"}, {"-d", "--data"}, {"--gt"}, {"-c", "--config"}, {"--occlusion"},
      {"-m", "--methods"}}));
  } catch (const std::exception & e) {
    std::cerr << "turn_dataset: " << e.what() << "\n"
              << "usage: turn_dataset --gt world_gt.yaml -c params.yaml -d turns.bin (--build raw.bin | "
      "[--occlusion none zbuffer mask] [-m none ransac mask2d])" << std::endl;
    return 1;
  }
  return 0;
}
