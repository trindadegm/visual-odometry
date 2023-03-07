import numpy as np

IDENTITY = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])

def camera(viewport_size, near, far):
    camera_a = (far + near)/(far - near)
    camera_b = (2*far*near)/(far - near)

    scale_w = viewport_size[1]/viewport_size[0]

    return np.array([
        [2*scale_w, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, camera_a, camera_b],
        [0.0, 0.0, -1.0, 0.0],
    ])

def mtranslate(x, y, z):
    return np.array([
        [1.0, 0.0, 0.0, x],
        [0.0, 1.0, 0.0, y],
        [0.0, 0.0, 1.0, z],
        [0.0, 0.0, 0.0, 1.0],
    ])

def mrot_x(angle):
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, np.cos(angle), -np.sin(angle), 0.0],
        [0.0, np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

def mrot_y(angle):
    return np.array([
        [np.cos(angle), 0.0, np.sin(angle), 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-np.sin(angle), 0.0, np.cos(angle), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

def mrot_z(angle):
    return np.array([
        [np.cos(angle), -np.sin(angle), 0.0, 0.0],
        [np.sin(angle), np.cos(angle), 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

def vnormalize(vector):
    length = np.sqrt((vector**2).sum())
    if abs(length) > 0.0:
        return vector/np.sqrt((vector**2).sum())
    else:
        return vector
