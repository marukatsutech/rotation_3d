""" Rotation in 3d """
import numpy as np
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import tkinter as tk
from tkinter import ttk
from matplotlib.patches import Circle
from scipy.spatial.transform import Rotation
import mpl_toolkits.mplot3d.art3d as art3d
from mpl_toolkits.mplot3d import proj3d

""" Global variables """

""" Animation control """
is_play = False

""" Axis vectors """
vector_x_axis = np.array([1., 0., 0.])
vector_y_axis = np.array([0., 1., 0.])
vector_z_axis = np.array([0., 0., 1.])

""" Other parameters """
theta_init_deg, phi_init_deg = 0., 0.
rot_velocity_x, rot_velocity_y, rot_velocity_z = 1., 1., 1.

""" Create figure and axes """
title_ax0 = "Rotation in 3D"
title_tk = title_ax0

x_min = -2.
x_max = 2.
y_min = -2.
y_max = 2.
z_min = -2.
z_max = 2.

fig = Figure()
ax0 = fig.add_subplot(111, projection='3d')
ax0.set_box_aspect((1, 1, 1))
ax0.grid()
ax0.set_title(title_ax0)
ax0.set_xlabel("x")
ax0.set_ylabel("y")
ax0.set_zlabel("t")
ax0.set_xlim(x_min, x_max)
ax0.set_ylim(y_min, y_max)
ax0.set_zlim(z_min, z_max)

""" Embed in Tkinter """
root = tk.Tk()
root.title(title_tk)
canvas = FigureCanvasTkAgg(fig, root)
canvas.get_tk_widget().pack(expand=True, fill="both")

toolbar = NavigationToolbar2Tk(canvas, root)
canvas.get_tk_widget().pack()

""" Global objects of Tkinter """
var_turn_op = tk.IntVar()

""" Classes and functions """


class Counter:
    def __init__(self, is3d=None, ax=None, xy=None, z=None, label=""):
        self.is3d = is3d if is3d is not None else False
        self.ax = ax
        self.x, self.y = xy[0], xy[1]
        self.z = z if z is not None else 0
        self.label = label

        self.count = 0

        if not is3d:
            self.txt_step = self.ax.text(self.x, self.y, self.label + str(self.count))
        else:
            self.txt_step = self.ax.text2D(self.x, self.y, self.label + str(self.count))
            self.xz, self.yz, _ = proj3d.proj_transform(self.x, self.y, self.z, self.ax.get_proj())
            self.txt_step.set_position((self.xz, self.yz))

    def count_up(self):
        self.count += 1
        self.txt_step.set_text(self.label + str(self.count))

    def reset(self):
        self.count = 0
        self.txt_step.set_text(self.label + str(self.count))

    def get(self):
        return self.count


def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi


def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


class Arrow3d:
    def __init__(self, ax, x, y, z, is_normalize, length, theta, phi, color, line_width, line_style, label):
        self.ax = ax
        self.x, self.y, self.z = x, y, z
        self.is_normalize = is_normalize
        self.r = length
        self.theta_init = theta
        self.phi_init = phi
        self.color = color
        self.line_width = line_width
        self.line_style = line_style
        self.label = label

        self.u, self.v, self.w = spherical_to_cartesian(self.r, self.theta_init, self.phi_init)

        if self.label != "":
            self.qvr = self.ax.quiver(self.x, self.y, self.z, self.u, self.v, self.w,
                                      length=1, color=self.color, normalize=self.is_normalize,
                                      linewidth=self.line_width, linestyle=self.line_style, label=self.label)
        else:
            self.qvr = self.ax.quiver(self.x, self.y, self.z, self.u, self.v, self.w,
                                      length=1, color=self.color, normalize=self.is_normalize, linewidth=self.line_width,
                                      linestyle=self.line_style)

        self.vector_init = np.array([self.u, self.v, self.w])
        self.is_rotate = True

    def _update_quiver(self):
        self.qvr.remove()
        if self.label != "":
            self.qvr = self.ax.quiver(self.x, self.y, self.z, self.u, self.v, self.w,
                                      length=1, color=self.color, normalize=self.is_normalize,
                                      linewidth=self.line_width, linestyle=self.line_style, label=self.label)
        else:
            self.qvr = self.ax.quiver(self.x, self.y, self.z, self.u, self.v, self.w,
                                      length=1, color=self.color, normalize=self.is_normalize,
                                      linewidth=self.line_width,
                                      linestyle=self.line_style)

    def rotate(self, angle, rotation_axis):
        if not self.is_rotate:
            return
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        rot_matrix = Rotation.from_rotvec(angle * rotation_axis)
        vector_rotated = rot_matrix.apply(self.vector_init)
        self.u, self.v, self.w = vector_rotated[0], vector_rotated[1], vector_rotated[2]
        self._update_quiver()

    def rotate2(self, angle_1st, angle_2nd, rotation_axis_1st, rotation_axis_2nd):
        if not self.is_rotate:
            return
        rotation_axis_1st = rotation_axis_1st / np.linalg.norm(rotation_axis_1st)
        rotation_axis_2nd = rotation_axis_2nd / np.linalg.norm(rotation_axis_2nd)
        rot_matrix_1st = Rotation.from_rotvec(angle_1st * rotation_axis_1st)
        rot_matrix_2nd = Rotation.from_rotvec(angle_2nd * rotation_axis_2nd)

        vector_rotated_1st = rot_matrix_1st.apply(self.vector_init)
        vector_rotated = rot_matrix_2nd.apply(vector_rotated_1st)

        self.u, self.v, self.w = vector_rotated[0], vector_rotated[1], vector_rotated[2]
        self._update_quiver()

    def rotate3(self, angle_1st, angle_2nd, angle_3rd, rotation_axis_1st, rotation_axis_2nd, rotation_axis_3rd):
        if not self.is_rotate:
            return
        rotation_axis_1st = rotation_axis_1st / np.linalg.norm(rotation_axis_1st)
        rotation_axis_2nd = rotation_axis_2nd / np.linalg.norm(rotation_axis_2nd)
        rotation_axis_3rd = rotation_axis_3rd / np.linalg.norm(rotation_axis_3rd)
        rot_matrix_1st = Rotation.from_rotvec(angle_1st * rotation_axis_1st)
        rot_matrix_2nd = Rotation.from_rotvec(angle_2nd * rotation_axis_2nd)
        rot_matrix_3rd = Rotation.from_rotvec(angle_3rd * rotation_axis_3rd)

        vector_rotated_1st = rot_matrix_1st.apply(self.vector_init)
        vector_rotated_2nd = rot_matrix_2nd.apply(vector_rotated_1st)
        vector_rotated = rot_matrix_3rd.apply(vector_rotated_2nd)

        self.u, self.v, self.w = vector_rotated[0], vector_rotated[1], vector_rotated[2]
        self._update_quiver()

    def set_rotate(self, flag):
        self.is_rotate = flag

    def set_direction(self, theta, phi):
        self.u, self.v, self.w = spherical_to_cartesian(self.r, theta, phi)
        self._update_quiver()

    def set_vector(self, u, v, w):
        self.u, self.v, self.w = u, v, w
        self._update_quiver()

    def set_theta_initial(self, theta):
        self.theta_init = theta
        self.u, self.v, self.w = spherical_to_cartesian(self.r, self.theta_init, self.phi_init)
        self.vector_init = np.array([self.u, self.v, self.w])
        self._update_quiver()

    def set_phi_initial(self, phi):
        self.phi_init = phi
        self.u, self.v, self.w = spherical_to_cartesian(self.r, self.theta_init, self.phi_init)
        self.vector_init = np.array([self.u, self.v, self.w])
        self._update_quiver()

    def get_vector(self):
        return np.array([self.u, self.v, self.w])


class ArrowPath:
    def __init__(self, ax, color, line_width, line_style, label):
        self.ax = ax
        self.color = color
        self.line_width = line_width
        self.line_style = line_style
        self.label = label

        self.x_path = []
        self.y_path = []
        self.z_path = []
        self.path, = self.ax.plot(np.array([]), np.array([]), np.array([]),
                                  color=self.color, linewidth=self.line_width, linestyle=self.line_style, label=self.label)

    def _re_plot(self):
        self.path.set_xdata(np.array(self.x_path))
        self.path.set_ydata(np.array(self.y_path))
        self.path.set_3d_properties(np.array(self.z_path))

    def add_xyz(self, x, y, z):
        self.x_path.append(x)
        self.y_path.append(y)
        self.z_path.append(z)
        self._re_plot()

    def reset(self):
        self.x_path = []
        self.y_path = []
        self.z_path = []
        self._re_plot()


def create_center_lines():
    line_axis_x = art3d.Line3D([0., 0.], [0., 0.], [z_min, z_max], color="gray", ls="-.", linewidth=1)
    ax0.add_line(line_axis_x)
    line_axis_y = art3d.Line3D([x_min, x_max], [0., 0.], [0., 0.], color="gray", ls="-.", linewidth=1)
    ax0.add_line(line_axis_y)
    line_axis_z = art3d.Line3D([0., 0.], [y_min, y_max], [0., 0.], color="gray", ls="-.", linewidth=1)
    ax0.add_line(line_axis_z)


def create_animation_control():
    frm_anim = ttk.Labelframe(root, relief="ridge", text="Animation", labelanchor="n")
    frm_anim.pack(side="left", fill=tk.Y)
    btn_play = tk.Button(frm_anim, text="Play/Pause", command=switch)
    btn_play.pack(side="left")
    btn_reset = tk.Button(frm_anim, text="Reset", command=reset)
    btn_reset.pack(side="left")


def set_theta_initial(theta):
    arrow.set_theta_initial(np.deg2rad(theta))
    arrow_path.reset()


def set_phi_initial(phi):
    arrow.set_phi_initial(np.deg2rad(phi))
    arrow_path.reset()


def set_vx(velocity):
    global rot_velocity_x
    rot_velocity_x = velocity


def set_vy(velocity):
    global rot_velocity_y
    rot_velocity_y = velocity


def set_vz(velocity):
    global rot_velocity_z
    rot_velocity_z = velocity


def create_parameter_setter():
    global var_turn_op
    frm_dir = ttk.Labelframe(root, relief="ridge", text="Initial direction", labelanchor='n')
    frm_dir.pack(side='left', fill=tk.Y)

    lbl_theta = tk.Label(frm_dir, text="Theta")
    lbl_theta.pack(side="left")

    var_theta = tk.StringVar(root)
    var_theta.set(str(theta_init_deg))
    spn_theta = tk.Spinbox(
        frm_dir, textvariable=var_theta, format="%.0f", from_=-360, to=360, increment=1,
        command=lambda: set_theta_initial(float(var_theta.get())), width=5
    )
    spn_theta.pack(side="left")

    lbl_phi = tk.Label(frm_dir, text="Phi")
    lbl_phi.pack(side="left")

    var_phi = tk.StringVar(root)
    var_phi.set(str(phi_init_deg))
    spn_phi = tk.Spinbox(
        frm_dir, textvariable=var_phi, format="%.0f", from_=-360, to=360, increment=1,
        command=lambda: set_phi_initial(float(var_phi.get())), width=5
    )
    spn_phi.pack(side="left")

    frm_v = ttk.Labelframe(root, relief="ridge", text="Rotation velocity", labelanchor='n')
    frm_v.pack(side='left', fill=tk.Y)

    lbl_vx = tk.Label(frm_v, text="x axis")
    lbl_vx.pack(side="left")

    var_vx = tk.StringVar(root)
    var_vx.set(str(rot_velocity_x))
    spn_vx = tk.Spinbox(
        frm_v, textvariable=var_vx, format="%.0f", from_=-10, to=10, increment=1,
        command=lambda: set_vx(float(var_vx.get())), width=5
    )
    spn_vx.pack(side="left")

    lbl_vy = tk.Label(frm_v, text="y axis")
    lbl_vy.pack(side="left")

    var_vy = tk.StringVar(root)
    var_vy.set(str(rot_velocity_y))
    spn_vy = tk.Spinbox(
        frm_v, textvariable=var_vy, format="%.0f", from_=-10, to=10, increment=1,
        command=lambda: set_vy(float(var_vy.get())), width=5
    )
    spn_vy.pack(side="left")

    lbl_vz = tk.Label(frm_v, text="z axis")
    lbl_vz.pack(side="left")

    var_vz = tk.StringVar(root)
    var_vz.set(str(rot_velocity_z))
    spn_vz = tk.Spinbox(
        frm_v, textvariable=var_vz, format="%.0f", from_=-10, to=10, increment=1,
        command=lambda: set_vz(float(var_vz.get())), width=5
    )
    spn_vz.pack(side="left")

    frm_turn = ttk.Labelframe(root, relief="ridge", text="Turn of rotation", labelanchor='n')
    frm_turn.pack(side='left', fill=tk.Y)

    var_turn_op = tk.IntVar()
    rd_op_xyz = tk.Radiobutton(frm_turn, text="x->y->z", value=1, variable=var_turn_op)
    rd_op_xyz.pack(side='left')

    rd_op_yzx = tk.Radiobutton(frm_turn, text="y->z->x", value=2, variable=var_turn_op)
    rd_op_yzx.pack(side='left')

    rd_op_zxy = tk.Radiobutton(frm_turn, text="z->x->y", value=3, variable=var_turn_op)
    rd_op_zxy.pack(side='left')

    var_turn_op.set(1)


def create_circle(ax, x, y, z, z_dir, edge_col, fill_flag, line_width, line_style, label):
    if label != "":
        c_spin_axis_guide = Circle((x, y), 1., ec=edge_col, fill=fill_flag,
                                   linewidth=line_width, linestyle=line_style, label=label)
    else:
        c_spin_axis_guide = Circle((x, y), 1., ec=edge_col, fill=fill_flag,
                                   linewidth=line_width, linestyle=line_style)
    ax.add_patch(c_spin_axis_guide)
    art3d.pathpatch_2d_to_3d(c_spin_axis_guide, z=z, zdir=z_dir)


def draw_static_diagrams():
    create_center_lines()
    create_circle(ax0, 0., 0., 0., "x", "gray", False, 0.5,
                  "--", "Light-sphere")
    create_circle(ax0, 0., 0., 0., "y", "gray", False, 0.5,
                  "--", "")
    create_circle(ax0, 0., 0., 0., "z", "gray", False, 0.5,
                  "--", "")


def update_diagrams():
    angle = np.deg2rad(cnt.get()) % (2. * np.pi)
    # arrow.set_direction(angle, angle)
    # arrow.rotate2(angle, angle, vector_x_axis, vector_y_axis)
    if var_turn_op.get() == 1:
        arrow.rotate3(rot_velocity_x * angle, rot_velocity_y * angle, rot_velocity_z * angle,
                      vector_x_axis, vector_y_axis, vector_z_axis)
    elif var_turn_op.get() == 2:
        arrow.rotate3(rot_velocity_y * angle, rot_velocity_z * angle, rot_velocity_x * angle,
                      vector_y_axis, vector_z_axis, vector_x_axis)
    else:
        arrow.rotate3(rot_velocity_z * angle, rot_velocity_x * angle, rot_velocity_y * angle,
                      vector_z_axis, vector_x_axis, vector_y_axis)
    point = arrow.get_vector()
    arrow_path.add_xyz(point[0], point[1], point[2])


def reset():
    global is_play
    is_play = False
    cnt.reset()
    arrow_path.reset()
    update_diagrams()


def switch():
    global is_play
    is_play = not is_play


def update(f):
    if is_play:
        cnt.count_up()
        update_diagrams()


""" main loop """
if __name__ == "__main__":
    cnt = Counter(ax=ax0, is3d=True, xy=np.array([x_min, y_max]), z=z_max, label="Step=")
    draw_static_diagrams()

    arrow = Arrow3d(ax0, 0., 0., 0., True, 1.,
                    np.deg2rad(theta_init_deg), np.deg2rad(phi_init_deg),
                    "Red", 2, "-", "Arrow")
    arrow_path = ArrowPath(ax0, "red", 1, "-", "Arrow path")

    # ax0.legend(loc='lower right', fontsize=8)

    create_animation_control()
    create_parameter_setter()

    anim = animation.FuncAnimation(fig, update, interval=100, save_count=100)
    root.mainloop()
