# Tóm tắt quá trình học NeRF — Assignment 3 (Volume Rendering & Surface Rendering)

Tài liệu này ghi lại toàn bộ pipeline mình đã implement, dựa trên README gốc của assignment, nhưng diễn giải lại theo đúng thứ tự tư duy: **từ 1 bức ảnh thô → ra được 1 scene 3D có thể render lại được**.

---

## Phần A — Neural Volume Rendering

### Bối cảnh đầu vào

Có 1 (hoặc nhiều) bức ảnh kích thước `W x H`, mỗi ảnh đi kèm 1 **camera đã biết trước pose** (vị trí, hướng nhìn, tiêu cự — do calibration hoặc COLMAP cung cấp). Mục tiêu cuối cùng: học ra 1 hàm số biểu diễn scene 3D, sao cho khi render lại từ đúng camera đó, ra ảnh khớp với ảnh gốc.

### Bước 1 — Pixel → NDC (Normalized Device Coordinate)

**File: `ray_utils.py` — `get_pixels_from_image`**

Từ ảnh `W x H`, mỗi pixel có tọa độ nguyên `(col, row)`. Bước đầu tiên là chuẩn hóa (normalize) 2 tọa độ này về khoảng `[-1, 1]`.

**Vì sao cần bước này:** các phép toán chiếu (projection) của camera trong PyTorch3D làm việc trên hệ tọa độ chuẩn hóa NDC, không phụ thuộc vào độ phân giải ảnh cụ thể — giúp công thức tổng quát cho mọi kích thước ảnh.

Kết quả: `xy_grid` shape `(W*H, 2)` — mỗi pixel có 1 cặp tọa độ NDC.

### Bước 2 — NDC → World coordinate → Ray origin + Ray direction

**File: `ray_utils.py` — `get_rays_from_pixels`**

**2a. Unproject — "chiếu ngược" pixel 2D thành điểm 3D:**

Vì 1 pixel không đủ thông tin để xác định 1 điểm 3D duy nhất (thiếu độ sâu), ta gán tạm `depth = 1` rồi dùng hàm có sẵn của camera để chiếu ngược ra 1 điểm 3D trong **world space**, nằm trên "mặt phẳng ảnh" tưởng tượng cách camera đúng 1 đơn vị theo hướng nhìn.

**2b. Tính Origin và Direction:**

```python
rays_o = camera.get_camera_center().repeat(N, 1)        
rays_d = F.normalize(image_plane_points - rays_o)         
```

Mỗi pixel giờ ứng với đúng 1 **tia (ray)** trong không gian 3D:


### Bước 3 — Sample điểm dọc theo tia (Point Sampling / Raymarching)

**File: `sampler.py` — `StratifiedSampler`**

Không thể evaluate scene tại vô số điểm dọc tia — chọn ra `n_pts_per_ray` mốc chia đều từ `near` đến `far`:

```python
z_vals = torch.linspace(near, far, n_pts_per_ray)
sample_points = rays_o.unsqueeze(1) + z_vals.view(1,-1,1) * rays_d.unsqueeze(1)
sample_lengths = z_vals  # khoảng cách tương ứng của từng điểm
```

Kết quả:
- `sample_points`: shape `(N_rays, N_pts, 3)` — tọa độ 3D của từng điểm sample
- `sample_lengths`: shape `(N_rays, N_pts, 1)` — khoảng cách (depth) tương ứng, dùng chung cho mọi tia

**Lưu ý quan trọng:** khác với "tìm giao điểm chính xác với vật thể" — cách này sample **đều đặn trên toàn bộ khoảng an toàn** `[near, far]`, để mô hình mật độ (density) tự học đâu là vùng có vật (density cao) và đâu là khoảng trống (density ≈ 0).

### Bước 4 — Evaluate scene tại từng điểm (Implicit Function)

Tại mỗi điểm sample, cần xác định: **density (σ) và màu (color) tại đây là bao nhiêu?**

**Positional Encoding** giúp mạng học được chi tiết tần số cao - các chi tiết bé xíu ở trong ảnh (fine detail) mà 1 MLP thường không học nổi từ tọa độ thô. 

### Bước 5 — Tổng hợp (Volume Rendering) — Q1.5

**File: `renderer.py` — `_compute_weights`, `_aggregate`**

Với `N_pts` cặp (density, color) dọc mỗi tia, cần "nén" lại thành 1 màu duy nhất.

**Transmittance**: xác suất tia sáng chiếu từ origin đến điểm thử i mà không bị chặn lại bởi bất kì điểm nào trước nó. Nếu tia sáng đi qua vùng không gian trống (mật độ $\sigma_j \approx 0$), transmittance sẽ dần về 1. Ngược lại, nếu đi qua vật thể đặc (mật độ $\sigma_j \approx 1$), transmittance sẽ dần về 0. Điều này giúp đảm bảo vật thể nằm sau vật thể đặc sẽ bị che khuất, không đóng góp màu sắc và pixel cuối cùng.

$$T_i = \prod_{j<i} e^{-\sigma_j\delta_j}$$

**Trọng số của từng điểm:** 

$$w_i = T_i \cdot \left(1 - e^{-\sigma_i \delta_i}\right)$$

**Tổng hợp màu và độ sâu:**

$$C = \sum_i w_i \cdot c_i \qquad D = \sum_i w_i \cdot z_i$$

 

### Bước 6 — Loss và tối ưu (Q2.2 / Q3.1)

Vì **toàn bộ pipeline từ Bước 1 đến Bước 5 đều khả vi (differentiable)**, gradient có thể lan truyền ngược từ "màu render sai bao nhiêu" tới tận trọng số của MLP (Q3) hoặc tham số hình học (Q2) — đây là "phép màu" cốt lõi của differentiable rendering.

### Sơ đồ tổng thể — Phần A

```
Ảnh (W x H) + Camera
    │  Bước 1 (Q1.3)
    ▼
Pixel (col,row) → tọa độ NDC (x,y)
    │  Bước 2 (Q1.3)
    ▼
Unproject → world point → Ray: origin (o) + direction (d)
    │  Bước 3 (Q1.4)
    ▼
N_pts điểm sample dọc tia: sample_points, sample_lengths
    │  Bước 4 (Q1/Q2: SDFVolume — hoặc Q3: NeRF MLP)
    ▼
Tại mỗi điểm: density σ, màu c
    │  Bước 5 (Q1.5)
    ▼
Transmittance T_i, trọng số w_i → tổng hợp: 1 màu + 1 depth / pixel
    │  ghép H x W pixel
    ▼
Ảnh render ra
    │  Bước 6 (Q2.2 / Q3.1)
    ▼
So sánh với ảnh thật → Loss (MSE) → Backprop → cập nhật tham số scene
```

---

## Phần B — Neural Surface Rendering

### Ý tưởng khác biệt so với Volume Rendering

Thay vì mô tả scene bằng "mật độ mờ khắp không gian" (density field), Phần B mô tả bằng **Signed Distance Function (SDF)** — hàm nhận vào 1 điểm 3D, trả về khoảng cách có dấu đến bề mặt gần nhất:

- `d > 0`: điểm nằm ngoài vật thể
- `d < 0`: điểm nằm trong vật thể
- `d = 0`: điểm nằm đúng trên bề mặt

→ Bề mặt = tập hợp điểm có `d = 0`, ranh giới **sắc nét** (khác hẳn density "mờ dần" của Phần A).

### Q5 — Sphere Tracing

**File: `renderer.py` — `sphere_tracing`**

Thay vì sample đều đặn nhiều điểm, dùng chính giá trị SDF để "nhảy" thẳng tới bề mặt — tại mỗi bước, SDF cho biết khoảng cách an toàn tối đa có thể march mà chắc chắn không "xuyên" qua vật.


### Q6 — Học Neural SDF từ Point Cloud

**File: `implicit.py` — `NeuralSurface`, `losses.py`**

**Point cloud loss** — ép SDF tại các điểm lấy từ point cloud (nằm trên bề mặt thật) tiến về 0:

$$\mathcal{L}_{\text{points}} = \frac{1}{N}\sum_i d(\mathbf{p}_i)^2$$


**Eikonal loss** — ràng buộc toán học bắt buộc của 1 SDF hợp lệ: độ dài gradient tại mọi điểm phải bằng 1:

$$\mathcal{L}_{\text{eikonal}} = \mathbb{E}_{\mathbf{x}}\left[\left(\|\nabla d(\mathbf{x})\|_2 - 1\right)^2\right]$$


