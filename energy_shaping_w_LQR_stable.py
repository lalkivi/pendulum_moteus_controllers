# This code plots the phase plot of the simple pendulum using the joint encoder
# during energy shaping swing up controller from the stable equilibrium and
# balance's upright using the LQR Controller)

import matplotlib; matplotlib.use("TkAgg")
import asyncio
import math as m
import moteus
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import csv
import os

# Pendulum Parameters
mass = 40 / 1000  # Mass in kg
gravity = 9.81  # Gravity in m/s^2
length = 75 / 1000  # Length in meters
max_motor_torque = 0.035  # Constrained Motor Torque Limit in Nm
K_e = 0.05 # Gain to amplify energy function
B = 0.0  # Damping term for dynamics
desired_angle = 180  # Upright position (angle in degrees)
energy_error_function_threshold = 0.045 # Threshold for switching to LQR

inertia = mass * length**2  # Moment of inertia

# State-space matrices for LQR control
A = np.array([[0, 1], [gravity / length, -B / inertia]])
B = np.array([[0], [1 / inertia]])
Q = np.diag([15.0, 1.5])  # Penalize position and velocity deviations
R = np.array([[1.5]])  # Penalize control effort

# Logging Control
is_logging = False  # Set to True to enable logging
log_folder = "logs"
log_filename = "energy_shaping_w_lqr_stable.csv"

# Plot Control
show_plots = True  # Set to False to disable showing plots

os.makedirs(log_folder, exist_ok=True)
log_filepath = os.path.join(log_folder, log_filename)

# Logging Variables
position_list = []
velocity_list = []
commanded_torque_list = []
desired_torque_list = []

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 9))

# Helper function to solve Algebraic Riccati Equation
def solve_are(A, B, Q, R, max_iter=1000, tol=1e-6, relaxation=0.00025):
    P = np.zeros_like(Q)
    for _ in range(max_iter):
        P_next = A.T @ P + P @ A - P @ B @ np.linalg.inv(R) @ B.T @ P + Q
        P_next = (1 - relaxation) * P + relaxation * P_next
        if np.allclose(P_next, P, atol=tol):
            return P_next
        P = P_next
    return P

P = solve_are(A, B, Q, R)
# Check for NaN in the solution
if np.isnan(P).any():
    print("Error: Riccati equation solver did not converge.")
    K_lqr = np.array([[0, 0]])
else:
    K_lqr = np.linalg.inv(R) @ B.T @ P
    print("LQR Gain Matrix:", K_lqr)

async def controller():
    c = moteus.Controller(id=1)
    await c.set_stop()
    prev_position = 0
    prev_velocity = 0
    try:
        while True:
            try:
                result = await c.query()
                position = (2 * m.pi * result.values[moteus.Register.POSITION])  # Position in radians
                velocity = (2 * m.pi * result.values[moteus.Register.VELOCITY])  # Velocity in radians/s
                commanded_torque = result.values[moteus.Register.TORQUE]

                # Desired Torque for swing up + lqr to stabilize unstable equilibrium
                kinetic_energy = (1 / 2) * mass * (length ** 2) * (velocity ** 2)  # Kinetic Energy of the Pendulum 1/2 mv^2
                potential_energy = mass * gravity * length * np.cos(position)  # Potential Energy of the Pendulum mglcos(theta)
                desired_energy_upright = mass * gravity * length * np.cos(np.deg2rad(desired_angle))  # Desired Potential Energy at the top

                energy_error_function = kinetic_energy - potential_energy - desired_energy_upright # Acts as a lyapunov_function

                if energy_error_function > 0.0:
                    desired_torque = 1 * K_e * velocity * energy_error_function  # Desired Torque in Nm
                    desired_torque = np.clip(desired_torque, -max_motor_torque, max_motor_torque)
                else:
                    desired_torque = -1 * K_e * velocity * energy_error_function  # Desired Torque in Nm
                    desired_torque = np.clip(desired_torque, -max_motor_torque, max_motor_torque)


                # Adding derivative of above function to estimate position for controller switching
                energy_error_function_derivative = -velocity * desired_torque  # Derivative of the lyapunov_function

                # Control Strategy Switching
                if (energy_error_function > 0) and (energy_error_function_derivative < 0) and (energy_error_function < energy_error_function_threshold):
                    result = await c.set_position(position=m.nan,
                                                  velocity=0.0,
                                                  maximum_torque=max_motor_torque,
                                                  feedforward_torque=desired_torque,
                                                  kp_scale=0.0,
                                                  kd_scale=0.0,
                                                  query=True)

                else:
                    print("Switching to LQR Stabilization")
                    des_vel = 0.1
                    kp_value = (K_lqr[0][0])
                    kd_value = (K_lqr[0][1])
                    desired_torque = 0.0
                    result = await c.set_position(position=m.nan,
                                                  velocity=des_vel,
                                                  maximum_torque=max_motor_torque,
                                                  stop_position=(desired_angle / 360),
                                                  kp_scale=kp_value,
                                                  kd_scale=kd_value,
                                                  query=True)

                position_list.append(np.rad2deg(position))
                velocity_list.append(np.rad2deg(velocity))
                commanded_torque_list.append(energy_error_function)
                desired_torque_list.append(energy_error_function_derivative)

            except asyncio.CancelledError:
                print("Controller loop cancelled.")
                break
            except Exception as e:
                print(f"Error in controller loop: {e}")

    finally:
        try:
            await c.set_stop()
        except Exception as e:
            print(f"Error while stopping controller: {e}")

async def main():
    try:
        await controller()
    finally:

        # Logging Variables
        if is_logging:
            # Save logged data to a .csv file
            with open(log_filepath, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Position (degrees)", "Velocity (degrees/sec)", "Commanded Torque (Nm)", "Desired Torque (Nm)"])
                for p, v, t, d_t in zip(position_list, velocity_list, commanded_torque_list, desired_torque_list):
                    writer.writerow([p, v, t, d_t])
            print(f"Data logged to {log_filepath}")

        # Plotting Variables
        if show_plots:
            fig.suptitle('Energy Shaping Swing-Up w LQR Controller Data', fontsize=12)

            ax1.plot(position_list, velocity_list)
            ax1.set_xlabel('Position (degrees)', fontsize=8)
            ax1.set_ylabel('Velocity (degrees/sec)', fontsize=8)
            ax1.set_title('Phase Plot', fontsize=10)
            ax1.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
            ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax1.set_xlim(-360, 360)

            ax2.plot(commanded_torque_list, label='Commanded Torque')
            ax2.plot(desired_torque_list, label='Desired Torque')
            ax2.set_xlabel('Time Steps', fontsize=8)
            ax2.set_ylabel('Torque (Nm)', fontsize=8)
            ax2.legend()
            ax2.set_title('Torque Plot', fontsize=10)

            ax3.plot(position_list)
            ax3.set_xlabel('Time Steps', fontsize=8)
            ax3.set_ylabel('Position (degrees)', fontsize=8)
            ax3.set_title('Position Plot', fontsize=10)

            ax4.plot(velocity_list)
            ax4.set_xlabel('Time Steps', fontsize=8)
            ax4.set_ylabel('Velocity (degrees/sec)', fontsize=8)
            ax4.set_title('Velocity Plot', fontsize=10)

            plt.tight_layout(rect=[0, 0, 1, 0.98])
            plt.show()

if __name__ == '__main__':
    asyncio.run(main())
