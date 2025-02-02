# This code plots the phase plot of the simple pendulum using the joint encoder
# during LQR controller balancing upright position.

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
B = 0.0  # Damping term for dynamics
desired_angle = 180 # desired angle in degrees
lqr_angle_threshold = 5 # threshold angle in degrees from the upright position where lqr controller will be applied

inertia = mass * length**2  # Moment of inertia

# State-space matrices
A = np.array([[0, 1], [gravity / length, -B / inertia]])
B = np.array([[0], [1 / inertia]])
Q = np.diag([10.0, 1.0])    # Reduce state cost matrix for smoother behavior
R = np.array([[1.0]])       # Increase control effort cost for smoother torque application

# Logging Control
is_logging = False  # Set to True to enable logging, False to disable
log_folder = "logs"  # Specify the folder to save the CSV
log_filename = "lqr_controller.csv"  # Specify the filename for logging

# Plot Control
show_plots = True  # Set to False to disable showing plots

# Ensure the folder exists
os.makedirs(log_folder, exist_ok=True)
log_filepath = os.path.join(log_folder, log_filename)

# Logging Variables
position_list = []
velocity_list = []
commanded_torque_list = []
desired_torque_list = []

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 9))

# Helper function to solve Algebraic Ricatti Equation
def solve_are(A, B, Q, R, max_iter=1000, tol=1e-6, relaxation=0.00025):
    P = np.zeros_like(Q)  # initial guess for P matrix
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
    K_lqr = np.array([[0, 0]])  # Set K_lqr to zero to avoid runtime errors if needed
else:
    K_lqr = np.linalg.inv(R) @ B.T @ P
    print("LQR Gain Matrix:", K_lqr)

async def controller():
    c = moteus.Controller(id=1)
    s = moteus.Stream(c)
    await c.set_stop()
    try:
        while True:
            try:
                result = await c.query()
                position = (2 * m.pi * result.values[moteus.Register.POSITION]) # Position in radians
                velocity = (2 * m.pi * result.values[moteus.Register.VELOCITY]) # Velcoity in radians/s
                commanded_torque = result.values[moteus.Register.TORQUE] # Commanded Torque in Nm

                # Desired Torque for lqr controller
                potential_energy = mass * gravity * length * np.cos(position)  # Potential Energy of the Pendulum mglcos(theta)
                energy_error_function_lqr_threshold = mass * gravity * length * np.cos(np.deg2rad(desired_angle - lqr_angle_threshold))
                position_error = (np.deg2rad(desired_angle) - position)

                # Implement dead zone around the upright position
                if (potential_energy >= energy_error_function_lqr_threshold):
                    des_vel = 0.0
                    kp_value = 0.0
                    kd_value = 0.0
                    desired_torque = 0.0
                else:
                    des_vel = 5.0
                    kp_value = (K_lqr[0][0])
                    kd_value = (K_lqr[0][1])
                    desired_torque_from_K = (K_lqr[0][0]) * position_error
                    desired_torque_from_B = (K_lqr[0][1]) * (des_vel - velocity)
                    desired_torque = (desired_torque_from_K + desired_torque_from_B)
                    desired_torque = np.clip(desired_torque, -max_motor_torque, max_motor_torque)


                result = await c.set_position(position=m.nan,
                                              velocity=des_vel,
                                              maximum_torque=max_motor_torque,
                                              stop_position=(desired_angle / 360),
                                              kp_scale=kp_value,
                                              kd_scale=kd_value,
                                              query=True)

                position_list.append(np.rad2deg(position))
                velocity_list.append(np.rad2deg(velocity))
                commanded_torque_list.append(commanded_torque)
                desired_torque_list.append(desired_torque)

            except asyncio.CancelledError:
                print("Controller loop cancelled.")
                break
            except Exception as e:
                print(f"Error in controller loop: {e}")

    finally:
        try:
            # Stop the controller safely
            await s.write_message(b'tel stop')
            await s.flush_read()
            await s.command(b'd stop')
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
            fig.suptitle('LQR Controller Data', fontsize=12)

            ax1.plot(position_list, velocity_list)
            ax1.set_xlabel('Position (degrees)', fontsize=8)
            ax1.set_ylabel('Velocity (degrees/sec)', fontsize=8)
            ax1.set_title('Phase Plot', fontsize=10)
            ax1.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
            ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax1.yaxis.get_offset_text().set_position((-0.01, 0))
            ax1.set_xlim(-360, 360)

            ax2.plot(commanded_torque_list)
            ax2.plot(desired_torque_list)
            ax2.set_xlabel('Time Steps', fontsize=8)
            ax2.set_ylabel('Torque (Nm)', fontsize=8)
            ax2.legend(['Commanded Torque', 'Desired Torque'])
            ax2.set_title('Torque Plot', fontsize=10)
            ax2.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
            ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax2.yaxis.get_offset_text().set_position((-0.01, 0))

            ax3.plot(position_list)
            ax3.set_xlabel('Time Steps', fontsize=8)
            ax3.set_ylabel('Position (degrees)', fontsize=8)
            ax3.set_title('Position Plot', fontsize=10)

            ax4.plot(velocity_list)
            ax4.set_xlabel('Time Steps', fontsize=8)
            ax4.set_ylabel('Velocity (degrees/sec)', fontsize=8)
            ax4.set_title('Velocity Plot', fontsize=10)
            ax4.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
            ax4.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax4.yaxis.get_offset_text().set_position((-0.01, 0))

            plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout with title spacing
            plt.show()

if __name__ == '__main__':
    asyncio.run(main())

# At the end of every script run the hard_reset_code to exit from motor braking