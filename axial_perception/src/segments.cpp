#include "axial_perception/segments.hpp"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace axial_perception
{

namespace
{

constexpr double kInf = std::numeric_limits<double>::infinity();

int max_label(const Labels & labels)
{
  return labels.empty() ? 0 : std::max(0, *std::max_element(labels.begin(), labels.end()));
}

Eigen::Matrix3d scatter(const Cloud & pts, const Eigen::Vector3d & c)
{
  const Cloud r = pts.colwise() - c;
  return r * r.transpose();
}

Eigen::ArrayXd dist2(const Cloud & pts, const Line & line)
{
  const Cloud r = pts.colwise() - line.c;
  return r.colwise().squaredNorm().array().transpose() -
         (line.d.transpose() * r).array().square().transpose();
}

std::vector<Line> random_lines(const Cloud & pts, int n, Rng & rng)
{
  std::uniform_int_distribution<int> pick(0, static_cast<int>(pts.cols()) - 1);
  std::vector<int> i(static_cast<size_t>(n));
  std::vector<int> j(static_cast<size_t>(n));
  for (auto & x : i) {
    x = pick(rng);
  }
  for (auto & x : j) {
    x = pick(rng);
  }
  std::vector<Line> out;
  for (size_t k = 0; k < i.size(); ++k) {
    const Eigen::Vector3d d = pts.col(j[k]) - pts.col(i[k]);
    const double norm = d.norm();
    if (norm > 1e-6) {
      out.push_back({pts.col(i[k]), d / norm});
    }
  }
  return out;
}

std::vector<std::vector<int>> knn(const Cloud & pts, int k)
{
  const int n = static_cast<int>(pts.cols());
  std::vector<std::vector<int>> out(static_cast<size_t>(n));
  std::vector<std::pair<double, int>> d(static_cast<size_t>(n > 0 ? n - 1 : 0));
  for (int i = 0; i < n; ++i) {
    size_t at = 0;
    for (int j = 0; j < n; ++j) {
      if (j != i) {
        d[at++] = {(pts.col(i) - pts.col(j)).squaredNorm(), j};
      }
    }
    std::partial_sort(d.begin(), d.begin() + k, d.end());
    for (int m = 0; m < k; ++m) {
      out[static_cast<size_t>(i)].push_back(d[static_cast<size_t>(m)].second);
    }
  }
  return out;
}

Eigen::Index argmin_row(const Eigen::ArrayXXd & a, Eigen::Index row)
{
  Eigen::Index best;
  a.row(row).minCoeff(&best);
  return best;
}

Labels argmin_rows(const Eigen::ArrayXXd & a)
{
  Labels out(static_cast<size_t>(a.rows()));
  for (Eigen::Index i = 0; i < a.rows(); ++i) {
    out[static_cast<size_t>(i)] = static_cast<int>(argmin_row(a, i));
  }
  return out;
}

std::vector<bool> prune(const Eigen::ArrayXXd & cost, double label_cost)
{
  const Eigen::Index n = cost.rows();
  std::vector<bool> keep(static_cast<size_t>(cost.cols()), true);
  auto kept = [&]() {return std::count(keep.begin(), keep.end(), true);};
  while (kept() > 1) {
    Indices idx;
    for (size_t j = 0; j < keep.size(); ++j) {
      if (keep[j]) {
        idx.push_back(static_cast<int>(j));
      }
    }
    std::vector<bool> used(idx.size(), false);
    used[0] = true;
    for (Eigen::Index i = 0; i < n; ++i) {
      size_t best = 0;
      for (size_t j = 1; j < idx.size(); ++j) {
        if (cost(i, idx[j]) < cost(i, idx[best])) {
          best = j;
        }
      }
      used[best] = true;
    }
    Indices live;
    for (size_t j = 0; j < idx.size(); ++j) {
      if (used[j]) {
        live.push_back(idx[j]);
      } else {
        keep[static_cast<size_t>(idx[j])] = false;
      }
    }
    if (kept() == 1) {
      break;
    }
    std::vector<double> saving(live.size(), 0.0);
    for (Eigen::Index i = 0; i < n; ++i) {
      size_t best = 0;
      double first = kInf;
      double second = kInf;
      for (size_t j = 0; j < live.size(); ++j) {
        const double c = cost(i, live[j]);
        if (c < first) {
          second = first;
          first = c;
          best = j;
        } else if (c < second) {
          second = c;
        }
      }
      saving[best] += second - first;
    }
    const auto low = std::min_element(saving.begin() + 1, saving.end());
    if (*low >= label_cost) {
      break;
    }
    keep[static_cast<size_t>(live[static_cast<size_t>(low - saving.begin())])] = false;
  }
  return keep;
}

}  // namespace

Line fit_line(const Cloud & pts)
{
  Line out;
  if (pts.cols() == 0) {
    return out;
  }
  out.c = pts.rowwise().mean();
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(scatter(pts, out.c));
  out.d = eig.eigenvectors().col(2);
  return out;
}

Eigen::Vector3d spread(const Cloud & pts, const Eigen::Vector3d & c)
{
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(scatter(pts, c), Eigen::EigenvaluesOnly);
  const Eigen::Vector3d ev = eig.eigenvalues().cwiseMax(0.0).cwiseSqrt();
  return {ev[2], ev[1], ev[0]};
}

Eigen::Vector3d axis_from_surface(const Eigen::Vector3d & c, const Eigen::Vector3d & d, double radius)
{
  const Eigen::Vector3d a = c - c.dot(d) * d;
  const double n = a.norm();
  return radius > 0 && n > 0 ? Eigen::Vector3d(c + radius * a / n) : c;
}

Eigen::ArrayXd line_distance(const Cloud & pts, const Line & line)
{
  const Cloud r = pts.colwise() - line.c;
  const Cloud perp = r - line.d * (line.d.transpose() * r);
  return perp.colwise().norm().array().transpose();
}

std::map<int, Indices> groups(const Labels & labels)
{
  std::map<int, Indices> out;
  for (size_t i = 0; i < labels.size(); ++i) {
    if (labels[i] > 0) {
      out[labels[i]].push_back(static_cast<int>(i));
    }
  }
  return out;
}

Labels gap_split(const Cloud & pts, const Labels & labels, double gap, int min_points)
{
  Labels out(labels.size(), 0);
  int n = 0;
  for (const auto & [k, sel] : groups(labels)) {
    const Cloud p = select(pts, sel);
    const Line line = fit_line(p);
    const Eigen::VectorXd t = line.d.transpose() * (p.colwise() - line.c);
    Indices order(sel.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {return t[a] < t[b];});
    size_t start = 0;
    for (size_t e = 1; e <= order.size(); ++e) {
      if (e == order.size() || (gap > 0 && t[order[e]] - t[order[e - 1]] > gap)) {
        if (static_cast<int>(e - start) >= min_points) {
          ++n;
          for (size_t m = start; m < e; ++m) {
            out[static_cast<size_t>(sel[static_cast<size_t>(order[m])])] = n;
          }
        }
        start = e;
      }
    }
  }
  return out;
}

Labels merge_lines(const Cloud & pts, Labels labels, double tol, double share)
{
  while (true) {
    const auto g = groups(labels);
    double best = share;
    std::pair<int, int> pair{0, 0};
    for (auto a = g.begin(); a != g.end(); ++a) {
      for (auto b = std::next(a); b != g.end(); ++b) {
        Indices sel;
        std::merge(a->second.begin(), a->second.end(), b->second.begin(), b->second.end(),
          std::back_inserter(sel));
        const Cloud p = select(pts, sel);
        const double fit = (line_distance(p, fit_line(p)) < tol).cast<double>().mean();
        if (fit >= best) {
          best = fit;
          pair = {a->first, b->first};
        }
      }
    }
    if (pair.first == 0) {
      return labels;
    }
    std::replace(labels.begin(), labels.end(), pair.second, pair.first);
  }
}

Labels finish(const Cloud & pts, const Labels & labels, double tol, int min_points, double gap)
{
  return gap_split(pts, merge_lines(pts, labels, tol), gap, min_points);
}

Labels drop_short(const Cloud & pts, const Labels & labels, double min_length)
{
  Labels out(labels.size(), 0);
  int n = 0;
  for (const auto & [k, sel] : groups(labels)) {
    const Cloud p = select(pts, sel);
    const Line line = fit_line(p);
    const Eigen::VectorXd t = line.d.transpose() * (p.colwise() - line.c);
    if (t.maxCoeff() - t.minCoeff() >= min_length) {
      ++n;
      for (int i : sel) {
        out[static_cast<size_t>(i)] = n;
      }
    }
  }
  return out;
}

Mst mst(const Cloud & pts)
{
  const Eigen::Index n = pts.cols();
  Mst out;
  out.parent.assign(static_cast<size_t>(n), 0);
  out.w.assign(static_cast<size_t>(n), 0.0);
  if (n == 0) {
    return out;
  }
  std::vector<bool> done(static_cast<size_t>(n), false);
  Eigen::ArrayXd dist = (pts.colwise() - pts.col(0)).colwise().norm().array().transpose();
  done[0] = true;
  dist[0] = kInf;
  out.order.push_back(0);
  for (Eigen::Index step = 1; step < n; ++step) {
    Eigen::Index j;
    dist.minCoeff(&j);
    out.w[static_cast<size_t>(j)] = dist[j];
    done[static_cast<size_t>(j)] = true;
    out.order.push_back(static_cast<int>(j));
    for (Eigen::Index i = 0; i < n; ++i) {
      if (!done[static_cast<size_t>(i)]) {
        const double nd = (pts.col(i) - pts.col(j)).norm();
        if (nd < dist[i]) {
          dist[i] = nd;
          out.parent[static_cast<size_t>(i)] = static_cast<int>(j);
        }
      }
    }
    dist[j] = kInf;
  }
  return out;
}

Indices longest_path(const Indices & nodes, const std::vector<int> & parent, const Cloud & pts)
{
  const size_t n = static_cast<size_t>(pts.cols());
  std::vector<bool> member(n, false);
  for (int v : nodes) {
    member[static_cast<size_t>(v)] = true;
  }
  std::vector<Indices> adj(n);
  for (int v : nodes) {
    const int p = parent[static_cast<size_t>(v)];
    if (member[static_cast<size_t>(p)] && p != v) {
      adj[static_cast<size_t>(v)].push_back(p);
      adj[static_cast<size_t>(p)].push_back(v);
    }
  }
  std::vector<double> dist(n);
  std::vector<int> prev(n);
  std::vector<bool> seen(n);
  auto farthest = [&](int src) {
      std::fill(seen.begin(), seen.end(), false);
      dist[static_cast<size_t>(src)] = 0.0;
      prev[static_cast<size_t>(src)] = -1;
      seen[static_cast<size_t>(src)] = true;
      int end = src;
      Indices stack{src};
      while (!stack.empty()) {
        const int u = stack.back();
        stack.pop_back();
        for (int v : adj[static_cast<size_t>(u)]) {
          if (!seen[static_cast<size_t>(v)]) {
            seen[static_cast<size_t>(v)] = true;
            dist[static_cast<size_t>(v)] = dist[static_cast<size_t>(u)] + (pts.col(u) - pts.col(v)).norm();
            prev[static_cast<size_t>(v)] = u;
            if (dist[static_cast<size_t>(v)] > dist[static_cast<size_t>(end)]) {
              end = v;
            }
            stack.push_back(v);
          }
        }
      }
      return end;
    };
  const int a = farthest(nodes.front());
  Indices path{farthest(a)};
  while (prev[static_cast<size_t>(path.back())] != -1) {
    path.push_back(prev[static_cast<size_t>(path.back())]);
  }
  return path;
}

Labels segmented_lsq(const Cloud & p, double penalty)
{
  const size_t m = static_cast<size_t>(p.cols());
  std::vector<Eigen::Vector3d> s1(m + 1, Eigen::Vector3d::Zero());
  std::vector<Eigen::Matrix3d> s2(m + 1, Eigen::Matrix3d::Zero());
  for (size_t i = 0; i < m; ++i) {
    const Eigen::Vector3d x = p.col(static_cast<Eigen::Index>(i));
    s1[i + 1] = s1[i] + x;
    s2[i + 1] = s2[i] + x * x.transpose();
  }
  std::vector<double> best(m + 1, 0.0);
  std::vector<size_t> arg(m + 1, 0);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig;
  for (size_t e = 1; e <= m; ++e) {
    double low = kInf;
    for (size_t i = 0; i < e; ++i) {
      const Eigen::Vector3d a = s1[e] - s1[i];
      const Eigen::Matrix3d cov = s2[e] - s2[i] - a * a.transpose() / static_cast<double>(e - i);
      eig.computeDirect(cov, Eigen::EigenvaluesOnly);
      const double v = best[i] + eig.eigenvalues()[0] + eig.eigenvalues()[1] + penalty;
      if (v < low) {
        low = v;
        arg[e] = i;
      }
    }
    best[e] = low;
  }
  Labels labels(m, 0);
  int k = 0;
  for (size_t e = m; e > 0; e = arg[e]) {
    ++k;
    std::fill(labels.begin() + static_cast<std::ptrdiff_t>(arg[e]),
      labels.begin() + static_cast<std::ptrdiff_t>(e), k);
  }
  return labels;
}

Labels no_split(const Cloud & pts, double, int min_points, double, Rng &)
{
  return Labels(static_cast<size_t>(pts.cols()), pts.cols() >= min_points ? 1 : 0);
}

Labels ransac(const Cloud & pts, double tol, int min_points, double gap, Rng & rng)
{
  constexpr int iters = 50;
  const size_t need = static_cast<size_t>(std::max(min_points, 1));
  Labels labels(static_cast<size_t>(pts.cols()), 0);
  Indices free(labels.size());
  std::iota(free.begin(), free.end(), 0);
  int next = 0;
  while (free.size() >= static_cast<size_t>(std::max(min_points, 2))) {
    const Cloud p = select(pts, free);
    const auto lines = random_lines(p, iters, rng);
    if (lines.empty()) {
      break;
    }
    Eigen::Index best_count = -1;
    Eigen::ArrayXd best;
    for (const auto & line : lines) {
      Eigen::ArrayXd d2 = dist2(p, line);
      const Eigen::Index count = (d2 < tol * tol).count();
      if (count > best_count) {
        best_count = count;
        best = std::move(d2);
      }
    }
    if (best_count < min_points) {
      break;
    }
    Indices inl;
    for (Eigen::Index j = 0; j < best.size(); ++j) {
      if (best[j] < tol * tol) {
        inl.push_back(static_cast<int>(j));
      }
    }
    const Eigen::ArrayXd d = line_distance(p, fit_line(select(p, inl)));
    if (static_cast<size_t>((d < tol).count()) < need) {
      break;
    }
    ++next;
    Indices rest;
    for (size_t j = 0; j < free.size(); ++j) {
      if (d[static_cast<Eigen::Index>(j)] < tol) {
        labels[static_cast<size_t>(free[j])] = next;
      } else {
        rest.push_back(free[j]);
      }
    }
    free = std::move(rest);
  }
  return finish(pts, labels, tol, min_points, gap);
}

Labels pearl(const Cloud & pts, double tol, int min_points, double gap, Rng & rng)
{
  constexpr int proposals = 30;
  constexpr double smooth = 0.3;
  constexpr double label_cost = 5.0;
  constexpr double deadzone = 0.7;
  constexpr int sweeps = 10;
  auto data = [&](const Eigen::ArrayXd & d2) -> Eigen::ArrayXd {
      return ((d2.max(0.0).sqrt() - deadzone * tol).max(0.0) / ((1 - deadzone) * tol)).square();
    };
  const Eigen::Index n = pts.cols();
  if (n < std::max(min_points, 3)) {
    return Labels(static_cast<size_t>(n), 0);
  }
  const Labels seed = ransac(pts, tol, min_points, 0, rng);
  std::vector<Line> lines;
  for (const auto & [s, sel] : groups(seed)) {
    lines.push_back(fit_line(select(pts, sel)));
  }
  for (const auto & l : random_lines(pts, proposals, rng)) {
    lines.push_back(l);
  }
  if (lines.empty()) {
    return Labels(static_cast<size_t>(n), 0);
  }
  const int k = std::min(6, static_cast<int>(n) - 1);
  const auto nbr = knn(pts, k);
  Eigen::ArrayXXd cost(n, static_cast<Eigen::Index>(lines.size()) + 1);
  cost.col(0).setOnes();
  for (size_t l = 0; l < lines.size(); ++l) {
    cost.col(static_cast<Eigen::Index>(l) + 1) = data(dist2(pts, lines[l]));
  }
  Labels labels = argmin_rows(cost);
  for (int sweep = 0; sweep < sweeps; ++sweep) {
    const std::vector<bool> keep = prune(cost, label_cost);
    std::vector<int> remap(keep.size(), -1);
    int cols = 0;
    for (size_t j = 0; j < keep.size(); ++j) {
      if (keep[j]) {
        remap[j] = cols++;
      }
    }
    Eigen::ArrayXXd pruned(n, cols);
    for (size_t j = 0; j < keep.size(); ++j) {
      if (keep[j]) {
        pruned.col(remap[j]) = cost.col(static_cast<Eigen::Index>(j));
      }
    }
    cost = std::move(pruned);
    for (auto & l : labels) {
      l = remap[static_cast<size_t>(l)];
    }
    Eigen::ArrayXXd total = cost + smooth * k;
    for (Eigen::Index i = 0; i < n; ++i) {
      for (int j : nbr[static_cast<size_t>(i)]) {
        const int l = labels[static_cast<size_t>(j)];
        if (l >= 0) {
          total(i, l) -= smooth;
        }
      }
    }
    const Labels next = argmin_rows(total);
    for (const auto & [l, sel] : groups(next)) {
      if (sel.size() >= 2) {
        cost.col(l) = data(line_distance(pts, fit_line(select(pts, sel))).square());
      }
    }
    if (next == labels) {
      break;
    }
    labels = next;
  }
  return finish(pts, labels, tol, min_points, gap);
}

Labels jlinkage(const Cloud & pts, double tol, int min_points, double gap, Rng & rng)
{
  constexpr int proposals = 100;
  constexpr int local = 10;
  const Eigen::Index n = pts.cols();
  if (n < std::max(min_points, 3)) {
    return Labels(static_cast<size_t>(n), 0);
  }
  const int k = std::min(local, static_cast<int>(n) - 1);
  const auto nbr = knn(pts, k);
  std::uniform_int_distribution<int> pick_point(0, static_cast<int>(n) - 1);
  std::uniform_int_distribution<int> pick_nbr(0, k - 1);
  std::vector<int> from(proposals);
  std::vector<int> col(proposals);
  for (auto & x : from) {
    x = pick_point(rng);
  }
  for (auto & x : col) {
    x = pick_nbr(rng);
  }
  std::vector<Line> lines;
  for (int p = 0; p < proposals; ++p) {
    const int i = from[static_cast<size_t>(p)];
    const Eigen::Vector3d d =
      pts.col(nbr[static_cast<size_t>(i)][static_cast<size_t>(col[static_cast<size_t>(p)])]) - pts.col(i);
    const double norm = d.norm();
    if (norm > 1e-6) {
      lines.push_back({pts.col(i), d / norm});
    }
  }
  Eigen::MatrixXd sets(n, static_cast<Eigen::Index>(lines.size()));
  for (size_t l = 0; l < lines.size(); ++l) {
    sets.col(static_cast<Eigen::Index>(l)) = (dist2(pts, lines[l]) < tol * tol).cast<double>().matrix();
  }
  Eigen::ArrayXd size = sets.rowwise().sum().array();
  const Eigen::MatrixXd inter = sets * sets.transpose();
  Eigen::ArrayXXd jac(n, n);
  for (Eigen::Index i = 0; i < n; ++i) {
    for (Eigen::Index j = 0; j < n; ++j) {
      jac(i, j) = i == j ? kInf : 1 - inter(i, j) / std::max(size[i] + size[j] - inter(i, j), 1.0);
    }
  }
  std::vector<int> owner(static_cast<size_t>(n));
  std::iota(owner.begin(), owner.end(), 0);
  std::vector<bool> alive(static_cast<size_t>(n), true);
  Eigen::ArrayXd rmin(n);
  std::vector<Eigen::Index> rarg(static_cast<size_t>(n));
  auto refresh = [&](Eigen::Index i) {rmin[i] = jac.row(i).minCoeff(&rarg[static_cast<size_t>(i)]);};
  for (Eigen::Index i = 0; i < n; ++i) {
    refresh(i);
  }
  while (true) {
    Eigen::Index a;
    rmin.minCoeff(&a);
    const Eigen::Index b = rarg[static_cast<size_t>(a)];
    if (rmin[a] >= 1) {
      break;
    }
    sets.row(a) = sets.row(a).cwiseProduct(sets.row(b));
    std::replace(owner.begin(), owner.end(), static_cast<int>(b), static_cast<int>(a));
    alive[static_cast<size_t>(b)] = false;
    jac.row(b).setConstant(kInf);
    jac.col(b).setConstant(kInf);
    rmin[b] = kInf;
    size[a] = sets.row(a).sum();
    const Eigen::VectorXd ia = sets * sets.row(a).transpose();
    Eigen::ArrayXd row(n);
    for (Eigen::Index j = 0; j < n; ++j) {
      row[j] = alive[static_cast<size_t>(j)] ?
        1 - ia[j] / std::max(size[j] + size[a] - ia[j], 1.0) : kInf;
    }
    row[a] = kInf;
    jac.row(a) = row.transpose();
    jac.col(a) = row;
    for (Eigen::Index j = 0; j < n; ++j) {
      const auto r = rarg[static_cast<size_t>(j)];
      if (alive[static_cast<size_t>(j)] && (j == a || r == a || r == b)) {
        refresh(j);
      }
    }
    for (Eigen::Index j = 0; j < n; ++j) {
      if (alive[static_cast<size_t>(j)] && jac(j, a) < rmin[j]) {
        rmin[j] = jac(j, a);
        rarg[static_cast<size_t>(j)] = a;
      }
    }
  }
  std::map<int, int> count;
  for (int o : owner) {
    ++count[o];
  }
  std::map<int, int> rank;
  for (const auto & [o, c] : count) {
    rank.emplace(o, static_cast<int>(rank.size()) + 1);
  }
  Labels labels(static_cast<size_t>(n));
  for (size_t i = 0; i < labels.size(); ++i) {
    labels[i] = count[owner[i]] >= min_points ? rank[owner[i]] : 0;
  }
  return finish(pts, labels, tol, min_points, gap);
}

Labels mst_dp(const Cloud & pts, double tol, int min_points, double gap, Rng &)
{
  constexpr double penalty = 5.0;
  const Eigen::Index n = pts.cols();
  Labels labels(static_cast<size_t>(n), 0);
  if (n < std::max(min_points, 3)) {
    return labels;
  }
  const Mst tree = mst(pts);
  std::vector<int> comp(static_cast<size_t>(n));
  std::iota(comp.begin(), comp.end(), 0);
  for (size_t m = 1; m < tree.order.size(); ++m) {
    const size_t v = static_cast<size_t>(tree.order[m]);
    if (tree.w[v] <= gap || gap <= 0) {
      comp[v] = comp[static_cast<size_t>(tree.parent[v])];
    }
  }
  std::map<int, Indices> by_comp;
  for (size_t v = 0; v < comp.size(); ++v) {
    by_comp[comp[v]].push_back(static_cast<int>(v));
  }
  for (const auto & [c, nodes] : by_comp) {
    if (static_cast<int>(nodes.size()) < min_points) {
      continue;
    }
    const Indices path = longest_path(nodes, tree.parent, pts);
    const Cloud along = select(pts, path);
    const Labels seg = segmented_lsq(along, penalty * tol * tol);
    const int offset = max_label(labels);
    for (int v : nodes) {
      Eigen::Index near;
      (along.colwise() - pts.col(v)).colwise().norm().minCoeff(&near);
      labels[static_cast<size_t>(v)] = seg[static_cast<size_t>(near)] + offset;
    }
  }
  for (const auto & [k, sel] : groups(labels)) {
    if (sel.size() >= 2) {
      const Cloud p = select(pts, sel);
      const Eigen::ArrayXd d = line_distance(p, fit_line(p));
      for (size_t j = 0; j < sel.size(); ++j) {
        if (d[static_cast<Eigen::Index>(j)] > tol) {
          labels[static_cast<size_t>(sel[j])] = 0;
        }
      }
    }
  }
  return finish(pts, labels, tol, min_points, gap);
}

const std::vector<std::string> & split_methods()
{
  static const std::vector<std::string> names{"none", "ransac", "pearl", "jlinkage", "mst_dp"};
  return names;
}

SplitFn split_method(const std::string & name)
{
  static const std::map<std::string, SplitFn> methods{
    {"none", no_split}, {"ransac", ransac}, {"pearl", pearl}, {"jlinkage", jlinkage},
    {"mst_dp", mst_dp}};
  const auto it = methods.find(name);
  if (it == methods.end()) {
    throw std::invalid_argument("unknown split method " + name);
  }
  return it->second;
}

Labels split_clusters(
  const Cloud & pts, const std::vector<int> & cls, const std::vector<int> & seg,
  const std::string & method, const std::map<int, double> & tols, int min_points,
  const std::map<int, double> & gaps, double min_length, uint64_t seed)
{
  const SplitFn split = split_method(method);
  Rng rng(seed);
  Labels line(static_cast<size_t>(pts.cols()), 0);
  std::map<int, Indices> by_seg;
  for (size_t i = 0; i < seg.size(); ++i) {
    by_seg[seg[i]].push_back(static_cast<int>(i));
  }
  for (const auto & [k, sel] : by_seg) {
    const int c = cls[static_cast<size_t>(sel.front())];
    const Labels lab = split(select(pts, sel), tols.at(c), min_points, gaps.at(c), rng);
    const int offset = max_label(line);
    for (size_t j = 0; j < sel.size(); ++j) {
      if (lab[j] > 0) {
        line[static_cast<size_t>(sel[j])] = lab[j] + offset;
      }
    }
  }
  return min_length > 0 ? drop_short(pts, line, min_length) : line;
}

}  // namespace axial_perception
