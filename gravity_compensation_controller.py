# This code plots the phase plot of the simple pendulum using the joint encoder
# during gravity compensation controller.

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
max_motor_torque = 0.035 # Constrained Motor Torque Limit in Nm (for max torque sin(theta) = 1)
K = 1.25 # Gain to compensate for uncertainities

# Logging Control
is_logging = False  # Set to True to enable logging, False to disable
log_folder = "logs"  # Specify the folder to save the CSV
log_filename = "gravity_compensation_controller.csv"  # Specify the filename for logging

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

                # Desired Torque for gravity compensation - mglsin(thetha)
                desired_torque = mass * gravity * length * np.sin(position) * K # Desired Torque in Nm

                result = await c.set_position(position=m.nan,
                                              velocity=0.0,
                                              maximum_torque=max_motor_torque,
                                              feedforward_torque=desired_torque,
                                              kp_scale=0.0,
                                              kd_scale=0.0,
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
            fig.suptitle('Gravity Compensation Controller Data', fontsize=12)

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
