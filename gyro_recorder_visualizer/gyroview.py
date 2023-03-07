import cv2
import linalg
import numpy as np
import pandas as pd
import sys
import time

VIEWPORT_SIZE = (800, 450)

CUBE_MODEL = np.array([
    [-1.0,-1.0,-1.0],
    [-1.0,-1.0, 1.0],
    [-1.0, 1.0, 1.0],

    [1.0, 1.0,-1.0],
    [-1.0,-1.0,-1.0],
    [-1.0, 1.0,-1.0],

    [1.0,-1.0, 1.0],
    [-1.0,-1.0,-1.0],
    [1.0,-1.0,-1.0],

    [1.0, 1.0,-1.0],
    [1.0,-1.0,-1.0],
    [-1.0,-1.0,-1.0],

    [-1.0,-1.0,-1.0],
    [-1.0, 1.0, 1.0],
    [-1.0, 1.0,-1.0],

    [1.0,-1.0, 1.0],
    [-1.0,-1.0, 1.0],
    [-1.0,-1.0,-1.0],

    [-1.0, 1.0, 1.0],
    [-1.0,-1.0, 1.0],
    [1.0,-1.0, 1.0],

    [1.0, 1.0, 1.0],
    [1.0,-1.0,-1.0],
    [1.0, 1.0,-1.0],

    [1.0,-1.0,-1.0],
    [1.0, 1.0, 1.0],
    [1.0,-1.0, 1.0],

    [1.0, 1.0, 1.0],
    [1.0, 1.0,-1.0],
    [-1.0, 1.0,-1.0],

    [1.0, 1.0, 1.0],
    [-1.0, 1.0,-1.0],
    [-1.0, 1.0, 1.0],

    [1.0, 1.0, 1.0],
    [-1.0, 1.0, 1.0],
    [1.0,-1.0, 1.0],
])

IDENTITY_TRANSFORM = linalg.IDENTITY

CAMERA_NEAR = 0.01
CAMERA_FAR = 20.0

CAMERA = linalg.camera(VIEWPORT_SIZE, CAMERA_NEAR, CAMERA_FAR)

AMBIENT_LIGTH_VECTOR = np.array([0.0, 0.0, -1.0]);

def main(input_filename, calibration_filename):
    data = pd.read_csv(input_filename, sep=',')
    timestamps = data.iloc[:, 0].to_numpy()
    accel_data = data.iloc[:, 1:4].to_numpy()
    gyro_data = data.iloc[:, 4:7].to_numpy()

    calibration_data = pd.read_csv(calibration_filename, sep=',')
    cal_gyro_x_mean = data.iloc[:, 4].mean()
    cal_gyro_y_mean = data.iloc[:, 5].mean()
    cal_gyro_z_mean = data.iloc[:, 6].mean()

    pos_x = 0.0
    pos_y = 0.0
    pos_z = -5.0

    rot_matrix = IDENTITY_TRANSFORM

    # render_frame_counter = 0
    # render_frame_time = 200

    last_ts = 0
    for i, ts in enumerate(timestamps):
        ts = int(ts/1000)
        frame_time = ts - last_ts

        print(f'Timestamp elapsed ms: {ts}ms')

        (gx, gy, gz) = gyro_data[i]
        # (gx, gy, gz) = (gx - cal_gyro_x_mean, gy - cal_gyro_y_mean, gz - cal_gyro_z_mean)

        # (ax, ay, az) = accel_data[i]

        angle_x = (gx/600) * np.pi/2
        angle_y = (gy/600) * np.pi/2
        angle_z = (gz/600) * np.pi/2

        rot_matrix = np.matmul(rot_matrix, linalg.mrot_x(angle_x))
        rot_matrix = np.matmul(rot_matrix, linalg.mrot_y(angle_y))
        rot_matrix = np.matmul(rot_matrix, linalg.mrot_z(angle_z))

        box_matrix = IDENTITY_TRANSFORM
        box_matrix = np.matmul(box_matrix, linalg.mtranslate(pos_x, pos_y, pos_z))

        box_matrix = np.matmul(box_matrix, rot_matrix)

        img = render_convex_volume(CUBE_MODEL, CAMERA, box_matrix)
        cv2.imshow('Gyro View', img)

        last_ts = ts

        if frame_time != 0 and cv2.waitKey(frame_time) == ord('q'):
            break

    cv2.destroyAllWindows()


def render_convex_volume(vertices, mview, mtransform):
    (columns, lines) = VIEWPORT_SIZE
    img = np.zeros((lines, columns, 3))

    def project_vertex(mproj, vertex):
        transformed_homomorphic = np.matmul(
            mproj,
            np.array([vertex[0], vertex[1], vertex[2], 1.0])
        )
        return transformed_homomorphic[0:3] / transformed_homomorphic[3]

    chunk_size = 3
    vertices = list(map(lambda v: project_vertex(mtransform, v), vertices))
    triangles = (vertices[i:i + chunk_size] for i in range(0, len(vertices), chunk_size))

    for (pre_a, pre_b, pre_c) in triangles:
        a = project_vertex(mview, pre_a)
        b = project_vertex(mview, pre_b)
        c = project_vertex(mview, pre_c)

        vec_ab = np.array(a) - np.array(b)
        vec_ac = np.array(a) - np.array(c)
        vec_out = np.cross(vec_ab, vec_ac)

        if vec_out[2] < 0.0:
            # We render this triangle
            ap = (a[0:2] * VIEWPORT_SIZE[1]).astype(int) + np.array(VIEWPORT_SIZE)//2
            bp = (b[0:2] * VIEWPORT_SIZE[1]).astype(int) + np.array(VIEWPORT_SIZE)//2
            cp = (c[0:2] * VIEWPORT_SIZE[1]).astype(int) + np.array(VIEWPORT_SIZE)//2

            vec_pre_ab = np.array(pre_a) - np.array(pre_b)
            vec_pre_ac = np.array(pre_a) - np.array(pre_c)
            vec_pre_out = np.cross(vec_pre_ab, vec_pre_ac)

            brightness = abs(np.dot(AMBIENT_LIGTH_VECTOR, linalg.vnormalize(vec_pre_out)))

            color = np.array([1.0, 0.0, 0.0]) * brightness
            color = (
                float(np.clip(color[0], 0.0, 1.0)),
                float(np.clip(color[1], 0.0, 1.0)),
                float(np.clip(color[2], 0.0, 1.0)),
            )

            try:
                cv2.fillConvexPoly(img, np.array([ap, bp, cp]), color=color)
            except Exception as e:
                print(e)
                print(ap, bp, cp)
                sys.exit(1)

    return img

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])