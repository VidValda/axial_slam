#include "axial_perception/projector.hpp"

#include <algorithm>
#include <stdexcept>

#include "axial_perception/occlusion.hpp"

namespace axial_perception
{

void check_occlusion_method(const std::string & method)
{
  if (method != "zbuffer" && method != "mask" && method != "none") {
    throw std::invalid_argument("unknown occlusion method " + method);
  }
}

MaskPoints mask_points(
  const Cloud & pts, const InstanceImage & img, const Intrinsics & K, const ProjectorParams & p)
{
  check_occlusion_method(p.occlusion_method);
  MaskPoints out;
  out.proj = project(pts, K, img.width, img.height, p.depth_min, p.depth_max);
  if (p.occlusion_method == "zbuffer") {
    std::vector<double> z;
    z.reserve(out.proj.size());
    for (int i : out.proj.idx) {
      z.push_back(pts(2, i));
    }
    out.proj = out.proj.select(
      front_mask(out.proj.u, out.proj.v, z, img.width, img.height, p.occlusion_window, p.occlusion_tol));
  } else if (p.occlusion_method == "mask") {
    std::vector<bool> on(out.proj.size());
    for (size_t i = 0; i < on.size(); ++i) {
      const int c = img.class_at(out.proj.u[i], out.proj.v[i]);
      on[i] = std::find(p.classes.begin(), p.classes.end(), c) != p.classes.end();
    }
    out.proj = out.proj.select(on);
    out.proj = out.proj.select(
      surface_mask(
        out.proj.u, out.proj.v, select(pts, out.proj.idx), img, p.occlusion_link, p.occlusion_radius,
        p.occlusion_behind));
  }
  out.labelled = label_points(out.proj, img, p.classes, p.min_points);
  return out;
}

}  // namespace axial_perception
