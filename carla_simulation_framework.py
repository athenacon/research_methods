#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Spawn NPCs into the simulation with taking data of every 20x,5y and 5 milliseconds interval."""

import glob
import os
import sys
import time
import math
import csv
# import pandas as pd NEVER IMPORT THIS - it messes up with data types

try:
    sys.path.append(glob.glob('/home/caramel/.conda/envs/carla/carla/PythonAPI/carla/dist/carla-0.9.8-py3.6-linux-x86_64.egg')[0])
except IndexError:
    pass

import carla

import argparse
import logging
import random

try:
    import queue
except ImportError:
    import Queue as queue

class CarlaSyncMode(object):

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 30)
        # self.delta_seconds = 0.1
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)

        for sensor in self.sensors[0]:
            make_queue(sensor.sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()

        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        # print(data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


class Camera(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self.surface = pygame.Surface((1280, 720))

        self._parent = parent_actor

        Attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid)]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)', {}],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)', {}],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)', {}],
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB Distorted',
                {'lens_circle_multiplier': '3.0',
                'lens_circle_falloff': '3.0',
                'chromatic_aberration_intensity': '0.5',
                'chromatic_aberration_offset': '0'}]]

        world = self._parent.get_world()

        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
                bp.set_attribute('image_size_x', f'{IM_WIDTH}')
                bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
                bp.set_attribute('fov', '110')
            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    # Activates sensors, and sets them to listen mode
    # Sensors indexes: 0: RGB, 1: Depth.raw, 2: Depth.Gray, 3: Depth.Log, 4: Semseg.raw, 5: Semseg.Cityscapespal
    # 6: Lidar, 7: RGB.Distorded
    def set_sensor(self, index, notify=True, force_respawn=False):
        self.index = index
        if self.sensor is None:
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[0][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[0][1])

        # circular reference.
        weak_self = weakref.ref(self)

        self.sensor.listen(lambda image: Camera._parse_image(weak_self, image))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return

        image.convert(self.sensors[self.index][1])
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        self.alt = 0.0
        self.t_s = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        # weak_self = weakref.ref(self)
        # self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude
        self.alt = event.altitude
        self.t_s = event.timestamp



# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================


class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        # weak_self = weakref.ref(self)
        # self.sensor.listen(lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)


# Function For Handling Sensor Data
def sensor_data(vehicle, gnss_sensor, imu_sensor):

    t = vehicle.get_transform()
    v = vehicle.get_velocity()
    c = vehicle.get_control()

    # Convert Compass Angles To Readings
    compass = math.degrees(imu_sensor.compass)
    heading = 'N' if compass > 270.5 or compass < 89.5 else ''
    heading += 'S' if 90.5 < compass < 269.5 else ''
    heading += 'E' if 0.5 < compass < 179.5 else ''
    heading += 'W' if 180.5 < compass < 359.5 else ''

    timestamp = gnss_sensor.timestamp
    limits = (-99.9, 99.9)

    accelerometer = (
        max(limits[0], min(limits[1], imu_sensor.accelerometer.x)),
        max(limits[0], min(limits[1], imu_sensor.accelerometer.y)),
        max(limits[0], min(limits[1], imu_sensor.accelerometer.z)))
    gyroscope = (
        max(limits[0], min(limits[1], math.degrees(imu_sensor.gyroscope.x))),
        max(limits[0], min(limits[1], math.degrees(imu_sensor.gyroscope.y))),
        max(limits[0], min(limits[1], math.degrees(imu_sensor.gyroscope.z))))

    sensordata = [
        # vehicle,
        round(timestamp, 3),
        math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2),
        round(compass, 2),
        round(accelerometer[0], 2),
        round(accelerometer[1], 2),
        round(accelerometer[2], 2),
        # round(gyroscope[0], 2),
        # round(gyroscope[1], 2),
        # round(gyroscope[2], 2),
        # round(t.location.x, 1),
        # round(t.location.y, 1),
        # round(t.location.z, 1),
        gnss_sensor.latitude,
        gnss_sensor.longitude,
        gnss_sensor.altitude,
        # c.gear,
        c.throttle,
        c.steer,
        # c.brake
    ]
    print(sensordata)
    return sensordata

def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-n', '--number-of-vehicles',
        metavar='N',
        default=30,
        type=int,
        help='number of vehicles (default: 10)')
    argparser.add_argument(
        '-w', '--number-of-walkers',
        metavar='W',
        default=50,
        type=int,
        help='number of walkers (default: 50)')
    argparser.add_argument(
        '--safe',
        action='store_true',
        help='avoid spawning vehicles prone to accidents')
    argparser.add_argument(
        '--filterv',
        metavar='PATTERN',
        default='vehicle.*',
        help='vehicles filter (default: "vehicle.*")')
    argparser.add_argument(
        '--filterw',
        metavar='PATTERN',
        default='walker.pedestrian.*',
        help='pedestrians filter (default: "walker.pedestrian.*")')
    argparser.add_argument(
        '-tm_p', '--tm_port',
        metavar='P',
        default=8000,
        type=int,
        help='port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    vehicles_list = []
    walkers_list = []
    all_id = []
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    try:

        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        world = client.get_world()
        # Get the current weather
        current_weather = world.get_weather()  # we get the previous weather bcs when I load another map it changes weather conitins
        world = client.load_world('Town02')  # simplest map
        # Set the weather again to maintain the same conditions
        world.set_weather(current_weather)

        synchronous_master = False

        if args.sync:
            settings = world.get_settings()
            traffic_manager.set_synchronous_mode(True)
            if not settings.synchronous_mode:
                synchronous_master = True
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
                world.apply_settings(settings)
            else:
                synchronous_master = False

        blueprints = world.get_blueprint_library().filter(args.filterv)
        blueprintsWalkers = world.get_blueprint_library().filter(args.filterw)

        if args.safe:
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
            blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
            blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('t2')]

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if args.number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif args.number_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, args.number_of_vehicles, number_of_spawn_points)
            args.number_of_vehicles = number_of_spawn_points

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # --------------
        # Spawn vehicles
        # --------------
        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= args.number_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True)))

        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)

        # -------------
        # View the simulation from above  # Pitch set to -90 for top-down view 200 200 250 -90
        spectator = world.get_spectator()
        transform = carla.Transform(carla.Location(x=100, y=200, z=190), carla.Rotation(pitch=-90))
        spectator.set_transform(transform)

        veh1 = world.get_actor(vehicles_list[0])
        Gnss1 = GnssSensor(veh1)
        IMU1 = IMUSensor(veh1)

        veh2 = world.get_actor(vehicles_list[1])
        Gnss2 = GnssSensor(veh2)
        IMU2 = IMUSensor(veh2)

        veh3 = world.get_actor(vehicles_list[2])
        Gnss3 = GnssSensor(veh3)
        IMU3 = IMUSensor(veh3)

        veh4 = world.get_actor(vehicles_list[3])
        Gnss4 = GnssSensor(veh4)
        IMU4 = IMUSensor(veh4)

        veh5 = world.get_actor(vehicles_list[4])
        Gnss5 = GnssSensor(veh5)
        IMU5 = IMUSensor(veh5)

        veh6 = world.get_actor(vehicles_list[5])
        Gnss6 = GnssSensor(veh6)
        IMU6 = IMUSensor(veh6)

        veh7 = world.get_actor(vehicles_list[6])
        Gnss7 = GnssSensor(veh7)
        IMU7 = IMUSensor(veh7)

        veh8 = world.get_actor(vehicles_list[7])
        Gnss8 = GnssSensor(veh8)
        IMU8 = IMUSensor(veh8)

        veh9 = world.get_actor(vehicles_list[8])
        Gnss9 = GnssSensor(veh9)
        IMU9 = IMUSensor(veh9)

        veh10 = world.get_actor(vehicles_list[8])
        Gnss10 = GnssSensor(veh10)
        IMU10 = IMUSensor(veh10)

        sensors = [Gnss1, IMU1, Gnss2, IMU2, Gnss3, IMU3, Gnss4, IMU4, Gnss5, IMU5, Gnss6, IMU6, Gnss7, IMU7, Gnss8,
                   IMU8, Gnss9, IMU9, Gnss10, IMU10]

        print('spawned %d vehicles press Ctrl+C to exit.' % (len(vehicles_list)))

        # example of how to use parameters
        traffic_manager.global_percentage_speed_difference(30.0)

        # while True:
        #     if args.sync and synchronous_master:
        #         world.tick()
        #     else:
        #         world.wait_for_tick()
        frame_counter = 0
        fps = 20
        csv_list = []
        csv_list1 = []
        csv_list2 = []
        csv_list3 = []
        csv_list4 = []
        csv_list5 = []
        csv_list6 = []
        csv_list7 = []
        csv_list8 = []
        csv_list9 = []
        csv_list10 = []
        with CarlaSyncMode(world, sensors, fps=fps) as sync_mode:
            while True:
                snapshot, gnss1, imu1, gnss2, imu2, gnss3, imu3, gnss4, imu4, gnss5, imu5, gnss6, imu6, gnss7, imu7, gnss8, imu8, gnss9, imu9, gnss10, imu10 = sync_mode.tick(
                timeout=100.0)
                # THIS IS FOR 9X 1Y - u can easily change the input output rows
                # if frame_counter == 0:
                #     print("ign")

                # elif frame_counter % 10 == 0:
                # elif frame_counter % 24 in [3, 8, 13, 18, 23]:
                if frame_counter % 25 < 20:  # First 19 frames are for 'x'
                    sensor_readings1 = sensor_data(veh1, gnss1, imu1)
                    sensor_readings1.insert(0, "x")
                    # csv_list.append(sensor_readings1)
                    csv_list1.append(sensor_readings1)

                    sensor_readings2 = sensor_data(veh2, gnss2, imu2)
                    sensor_readings2.insert(0, "x")
                    # csv_list.append(sensor_readings2)
                    csv_list2.append(sensor_readings2)

                    sensor_readings3 = sensor_data(veh3, gnss3, imu3)
                    sensor_readings3.insert(0, "x")
                    # csv_list.append(sensor_readings3)
                    csv_list3.append(sensor_readings3)

                    sensor_readings4 = sensor_data(veh4, gnss4, imu4)
                    sensor_readings4.insert(0, "x")
                    # csv_list.append(sensor_readings4)
                    csv_list4.append(sensor_readings4)

                    sensor_readings5 = sensor_data(veh5, gnss5, imu5)
                    sensor_readings5.insert(0, "x")
                    # csv_list.append(sensor_readings5)
                    csv_list5.append(sensor_readings5)

                    sensor_readings6 = sensor_data(veh6, gnss6, imu6)
                    sensor_readings6.insert(0, "x")
                    # csv_list.append(sensor_readings6)
                    csv_list6.append(sensor_readings6)

                    sensor_readings7 = sensor_data(veh7, gnss7, imu7)
                    sensor_readings7.insert(0, "x")
                    # csv_list.append(sensor_readings7)
                    csv_list7.append(sensor_readings7)

                    sensor_readings8 = sensor_data(veh8, gnss8, imu8)
                    sensor_readings8.insert(0, "x")
                    # csv_list.append(sensor_readings8)
                    csv_list8.append(sensor_readings8)

                    sensor_readings9 = sensor_data(veh9, gnss9, imu9)
                    sensor_readings9.insert(0, "x")
                    # csv_list.append(sensor_readings9)
                    csv_list9.append(sensor_readings9)

                    sensor_readings10 = sensor_data(veh10, gnss10, imu10)
                    sensor_readings10.insert(0, "x")
                    # csv_list.append(sensor_readings10)
                    csv_list10.append(sensor_readings10)

                else:
                    sensor_readings1 = sensor_data(veh1, gnss1, imu1)
                    sensor_readings1.insert(0, "y")
                    # csv_list.append(sensor_readings1)
                    csv_list1.append(sensor_readings1)

                    sensor_readings2 = sensor_data(veh2, gnss2, imu2)
                    sensor_readings2.insert(0, "y")
                    # csv_list.append(sensor_readings2)
                    csv_list2.append(sensor_readings2)

                    sensor_readings3 = sensor_data(veh3, gnss3, imu3)
                    sensor_readings3.insert(0, "y")
                    # csv_list.append(sensor_readings3)
                    csv_list3.append(sensor_readings3)

                    sensor_readings4 = sensor_data(veh4, gnss4, imu4)
                    sensor_readings4.insert(0, "y")
                    # csv_list.append(sensor_readings4)
                    csv_list4.append(sensor_readings4)

                    sensor_readings5 = sensor_data(veh5, gnss5, imu5)
                    sensor_readings5.insert(0, "y")
                    # csv_list.append(sensor_readings5)
                    csv_list5.append(sensor_readings5)

                    sensor_readings6 = sensor_data(veh6, gnss6, imu6)
                    sensor_readings6.insert(0, "y")
                    # csv_list.append(sensor_readings6)
                    csv_list6.append(sensor_readings6)

                    sensor_readings7 = sensor_data(veh7, gnss7, imu7)
                    sensor_readings7.insert(0, "y")
                    # csv_list.append(sensor_readings7)
                    csv_list7.append(sensor_readings7)

                    sensor_readings8 = sensor_data(veh8, gnss8, imu8)
                    sensor_readings8.insert(0, "y")
                    # csv_list.append(sensor_readings8)
                    csv_list8.append(sensor_readings8)

                    sensor_readings9 = sensor_data(veh9, gnss9, imu9)
                    sensor_readings9.insert(0, "y")
                    # csv_list.append(sensor_readings9)
                    csv_list9.append(sensor_readings9)

                    sensor_readings10 = sensor_data(veh10, gnss10, imu10)
                    sensor_readings10.insert(0, "y")
                    csv_list.append(sensor_readings10)
                    csv_list10.append(sensor_readings10)

                frame_counter += 1

        # csv_list.append(sensor_readings1, sensor_readings2,sensor_readings3,sensor_readings4,sensor_readings5,sensor_readings6,
        #                 sensor_readings7,sensor_readings8,sensor_readings9,sensor_readings10)
    finally:

        # csv_list.append(csv_list1)
        # csv_list.append(csv_list2)
        save_measurement = True
        storage_path = '/home/cmax/Desktop/athena/KIOS/obtained_data/new_data/'
        headers = "a55, a22,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12"


        if save_measurement:
            savename = 'perf_world_new_data'
            savename1 = 'veh1_scen_001'
            savename2 = 'veh1_scen_002'
            savename3 = 'veh1_scen_003'
            savename4 = 'veh1_scen_004'
            savename5 = 'veh1_scen_005'
            savename6 = 'veh1_scen_006'
            savename7 = 'veh1_scen_007'
            savename8 = 'veh1_scen_008'
            savename9 = 'veh1_scen_009'
            savename10 = 'veh1_scen_010'

            filename = storage_path + savename + '.csv'
            filename1 = storage_path + savename1 + '.csv'

            filename2 = storage_path + savename2 + '.csv'
            filename3 = storage_path + savename3 + '.csv'
            filename4 = storage_path + savename4 + '.csv'
            filename5 = storage_path + savename5 + '.csv'
            filename6 = storage_path + savename6 + '.csv'
            filename7 = storage_path + savename7 + '.csv'
            filename8 = storage_path + savename8 + '.csv'
            filename9 = storage_path + savename9 + '.csv'
            filename10 = storage_path + savename10 + '.csv'
            # Check if file already exists
            file_exists = os.path.isfile(filename1)
            headers = ["Vehicle", "Time (s)", "Speed", "Compass", "Accelerometer_x", "Accelerometer_y",
                                     "Accelerometer_z", "GNSS_latitude", "GNSS_longitude",
                                     "GNSS_altitude", "Control_throttle", "Control_steer"]
            #
            # with open(filename, 'a' if file_exists else 'w', newline='') as file:
            #     writer = csv.writer(file)
            #
            #     for sublist in csv_list:
            #         # This will flatten the sublist and write it as a single row
            #         flattened = [item for sublist in sublist for item in sublist]
            #         writer.writerow(flattened)
            #
            # with open(filename, 'a' if file_exists else 'w', newline='') as f:
            #     # print(f'Appending to file: {filename}')  # To check if we're in the correct condition
            #     writer = csv.writer(f)
            #
            #     # If file doesn't exist, write the header
            #     if not file_exists:
            #         writer.writerow(["Vehicle", "Time (s)", "Speed", "Compass", "Accelerometer_x", "Accelerometer_y",
            #                          "Accelerometer_z", "GNSS_latitude", "GNSS_longitude",
            #                          "GNSS_altitude", "Control_throttle", "Control_steer"])
            #
            #     # Write the data
            #     # print(f'Appending data: {csv_list}')  # To check the data before appending
            #     writer.writerows(csv_list)

            with open(filename1, 'a' if file_exists else 'w', newline='') as f:
                # print(f'Appending to file: {filename}')  # To check if we're in the correct condition
                writer = csv.writer(f)

                # If file doesn't exist, write the header
                if not file_exists:
                    writer.writerow(["Vehicle", "Time (s)", "Speed", "Compass", "Accelerometer_x", "Accelerometer_y",
                                     "Accelerometer_z", "GNSS_latitude", "GNSS_longitude",
                                     "GNSS_altitude", "Control_throttle", "Control_steer"])

                # Write the data
                # print(f'Appending data: {csv_list1}')  # To check the data before appending
                writer.writerows(csv_list1)

            with open(filename2, 'a' if file_exists else 'w', newline='') as f:
                # print(f'Appending to file: {filename}')  # To check if we're in the correct condition
                writer = csv.writer(f)

                # If file doesn't exist, write the header
                if not file_exists:
                    writer.writerow(["Vehicle", "Time (s)", "Speed", "Compass", "Accelerometer_x", "Accelerometer_y",
                                     "Accelerometer_z", "GNSS_latitude", "GNSS_longitude",
                                     "GNSS_altitude", "Control_throttle", "Control_steer"])

                # Write the data
                # print(f'Appending data: {csv_list1}')  # To check the data before appending
                writer.writerows(csv_list2)

            with open(filename3, 'a' if file_exists else 'w', newline='') as f:
                # print(f'Appending to file: {filename}')  # To check if we're in the correct condition
                writer = csv.writer(f)

                # If file doesn't exist, write the header
                if not file_exists:
                    writer.writerow(["Vehicle", "Time (s)", "Speed", "Compass", "Accelerometer_x", "Accelerometer_y",
                                     "Accelerometer_z", "GNSS_latitude", "GNSS_longitude",
                                     "GNSS_altitude", "Control_throttle", "Control_steer"])

                # Write the data
                # print(f'Appending data: {csv_list1}')  # To check the data before appending
                writer.writerows(csv_list3)

            with open(filename4, 'a' if file_exists else 'w', newline='') as f:
                # print(f'Appending to file: {filename}')  # To check if we're in the correct condition
                writer = csv.writer(f)

                # If file doesn't exist, write the header
                if not file_exists:
                    writer.writerow(["Vehicle", "Time (s)", "Speed", "Compass", "Accelerometer_x", "Accelerometer_y",
                                     "Accelerometer_z", "GNSS_latitude", "GNSS_longitude",
                                     "GNSS_altitude", "Control_throttle", "Control_steer"])

                # Write the data
                # print(f'Appending data: {csv_list1}')  # To check the data before appending
                writer.writerows(csv_list4)

            with open(filename5, 'a' if file_exists else 'w', newline='') as f:
                # print(f'Appending to file: {filename}')  # To check if we're in the correct condition
                writer = csv.writer(f)

                # If file doesn't exist, write the header
                if not file_exists:
                    writer.writerow(["Vehicle", "Time (s)", "Speed", "Compass", "Accelerometer_x", "Accelerometer_y",
                                     "Accelerometer_z", "GNSS_latitude", "GNSS_longitude",
                                     "GNSS_altitude", "Control_throttle", "Control_steer"])

                # Write the data
                # print(f'Appending data: {csv_list1}')  # To check the data before appending
                writer.writerows(csv_list5)
            with open(filename6, 'a' if file_exists else 'w', newline='') as f:
                # print(f'Appending to file: {filename}')  # To check if we're in the correct condition
                writer = csv.writer(f)

                # If file doesn't exist, write the header
                if not file_exists:
                    writer.writerow(["Vehicle", "Time (s)", "Speed", "Compass", "Accelerometer_x", "Accelerometer_y",
                                     "Accelerometer_z", "GNSS_latitude", "GNSS_longitude",
                                     "GNSS_altitude", "Control_throttle", "Control_steer"])

                # Write the data
                # print(f'Appending data: {csv_list1}')  # To check the data before appending
                writer.writerows(csv_list6)
            with open(filename7, 'a' if file_exists else 'w', newline='') as f:
                # print(f'Appending to file: {filename}')  # To check if we're in the correct condition
                writer = csv.writer(f)

                # If file doesn't exist, write the header
                if not file_exists:
                    writer.writerow(["Vehicle", "Time (s)", "Speed", "Compass", "Accelerometer_x", "Accelerometer_y",
                                     "Accelerometer_z", "GNSS_latitude", "GNSS_longitude",
                                     "GNSS_altitude", "Control_throttle", "Control_steer"])

                # Write the data
                # print(f'Appending data: {csv_list1}')  # To check the data before appending
                writer.writerows(csv_list7)

            with open(filename8, 'a' if file_exists else 'w', newline='') as f:
                # print(f'Appending to file: {filename}')  # To check if we're in the correct condition
                writer = csv.writer(f)

                # If file doesn't exist, write the header
                if not file_exists:
                    writer.writerow(["Vehicle", "Time (s)", "Speed", "Compass", "Accelerometer_x", "Accelerometer_y",
                                     "Accelerometer_z", "GNSS_latitude", "GNSS_longitude",
                                     "GNSS_altitude", "Control_throttle", "Control_steer"])

                # Write the data
                # print(f'Appending data: {csv_list1}')  # To check the data before appending
                writer.writerows(csv_list8)
            with open(filename9, 'a' if file_exists else 'w', newline='') as f:
                # print(f'Appending to file: {filename}')  # To check if we're in the correct condition
                writer = csv.writer(f)

                # If file doesn't exist, write the header
                if not file_exists:
                    writer.writerow(["Vehicle", "Time (s)", "Speed", "Compass", "Accelerometer_x", "Accelerometer_y",
                                     "Accelerometer_z", "GNSS_latitude", "GNSS_longitude",
                                     "GNSS_altitude", "Control_throttle", "Control_steer"])

                # Write the data
                # print(f'Appending data: {csv_list1}')  # To check the data before appending
                writer.writerows(csv_list9)

            with open(filename10, 'a' if file_exists else 'w', newline='') as f:
                # print(f'Appending to file: {filename}')  # To check if we're in the correct condition
                writer = csv.writer(f)

                # If file doesn't exist, write the header
                if not file_exists:
                    writer.writerow(["Vehicle", "Time (s)", "Speed", "Compass", "Accelerometer_x", "Accelerometer_y",
                                     "Accelerometer_z", "GNSS_latitude", "GNSS_longitude",
                                     "GNSS_altitude", "Control_throttle", "Control_steer"])

                # Write the data
                # print(f'Appending data: {csv_list1}')  # To check the data before appending
                writer.writerows(csv_list10)
        #
        #     df = pd.DataFrame(csv_list)  # my data is a list of lists
        #     df1 = pd.DataFrame(csv_list1)  # my data is a list of lists
        #     df2 = pd.DataFrame(csv_list2)  # my data is a list of lists
        #     df3 = pd.DataFrame(csv_list3)  # my data is a list of lists
        #     df4 = pd.DataFrame(csv_list4)  # my data is a list of lists
        #     df5 = pd.DataFrame(csv_list5)  # my data is a list of lists
        #     df6 = pd.DataFrame(csv_list6)  # my data is a list of lists
        #     df7 = pd.DataFrame(csv_list7)  # my data is a list of lists
        #     df8 = pd.DataFrame(csv_list8)  # my data is a list of lists
        #     df9 = pd.DataFrame(csv_list9)  # my data is a list of lists
        #     df10 = pd.DataFrame(csv_list10)  # my data is a list of lists
        #
        #     filenames = [filename1, filename2, filename3, filename4, filename5, filename6, filename7, filename8,
        #                  filename9, filename10]
        #     # Create directories if they don't exist
        #     os.makedirs(os.path.dirname(filename1), exist_ok=True)
        #
        #     # Check if file exists to avoid writing headers multiple times
        #     if not os.path.isfile(filename):
        #         df.to_csv(filename, index=False,
        #                   header=["Vehicle", "Time (s)", "Speed", "Compass", "Accelerometer_x", "Accelerometer_y",
        #                           "Accelerometer_z", "GNSS_latitude", "GNSS_longitude",
        #                           "GNSS_altitude", "Control_throttle", "Control_steer"])
        #     else:  # else it exists so append without writing the header
        #         df.to_csv(filename, mode='a', index=False, header=False)
        #
        #     # Check if file exists to avoid writing headers multiple times
        #     if not os.path.isfile(filename1):
        #         df1.to_csv(filename1, index=False,
        #                    header=["Vehicle", "Time (s)", "Speed", "Compass", "Accelerometer_x", "Accelerometer_y",
        #                            "Accelerometer_z", "GNSS_latitude", "GNSS_longitude",
        #                            "GNSS_altitude", "Control_throttle", "Control_steer"])
        #     else:  # else it exists so append without writing the header
        #         df1.to_csv(filename1, mode='a', index=False, header=False)
        #
        #         # Check if file exists to avoid writing headers multiple times
        #     if not os.path.isfile(filename2):
        #         df2.to_csv(filename2, index=False,
        #                    header=["Vehicle", "Time (s)", "Speed", "Compass", "Accelerometer_x", "Accelerometer_y",
        #                            "Accelerometer_z", "GNSS_latitude", "GNSS_longitude",
        #                            "GNSS_altitude", "Control_throttle", "Control_steer"])
        #     else:  # else it exists so append without writing the header
        #         df2.to_csv(filename2, mode='a', index=False, header=False)
        #
        #         # Check if file exists to avoid writing headers multiple times
        #     if not os.path.isfile(filename3):
        #         df3.to_csv(filename3, index=False,
        #                    header=["Vehicle", "Time (s)", "Speed", "Compass", "Accelerometer_x", "Accelerometer_y",
        #                            "Accelerometer_z", "GNSS_latitude", "GNSS_longitude",
        #                            "GNSS_altitude", "Control_throttle", "Control_steer"])
        #     else:  # else it exists so append without writing the header
        #         df3.to_csv(filename3, mode='a', index=False, header=False)
        #
        #         # Check if file exists to avoid writing headers multiple times
        #     if not os.path.isfile(filename4):
        #         df4.to_csv(filename4, index=False,
        #                    header=["Vehicle", "Time (s)", "Speed", "Compass", "Accelerometer_x", "Accelerometer_y",
        #                            "Accelerometer_z", "GNSS_latitude", "GNSS_longitude",
        #                            "GNSS_altitude", "Control_throttle", "Control_steer"])
        #     else:  # else it exists so append without writing the header
        #         df4.to_csv(filename4, mode='a', index=False, header=False)
        #
        #         # Check if file exists to avoid writing headers multiple times
        #     if not os.path.isfile(filename5):
        #         df5.to_csv(filename5, index=False,
        #                    header=["Vehicle", "Time (s)", "Speed", "Compass", "Accelerometer_x", "Accelerometer_y",
        #                            "Accelerometer_z", "GNSS_latitude", "GNSS_longitude",
        #                            "GNSS_altitude", "Control_throttle", "Control_steer"])
        #     else:  # else it exists so append without writing the header
        #         df5.to_csv(filename5, mode='a', index=False, header=False)
        #
        #         # Check if file exists to avoid writing headers multiple times
        #     if not os.path.isfile(filename6):
        #         df6.to_csv(filename6, index=False,
        #                    header=["Vehicle", "Time (s)", "Speed", "Compass", "Accelerometer_x", "Accelerometer_y",
        #                            "Accelerometer_z", "GNSS_latitude", "GNSS_longitude",
        #                            "GNSS_altitude", "Control_throttle", "Control_steer"])
        #     else:  # else it exists so append without writing the header
        #         df6.to_csv(filename6, mode='a', index=False, header=False)
        #
        #         # Check if file exists to avoid writing headers multiple times
        #     if not os.path.isfile(filename7):
        #         df7.to_csv(filename7, index=False,
        #                    header=["Vehicle", "Time (s)", "Speed", "Compass", "Accelerometer_x", "Accelerometer_y",
        #                            "Accelerometer_z", "GNSS_latitude", "GNSS_longitude",
        #                            "GNSS_altitude", "Control_throttle", "Control_steer"])
        #     else:  # else it exists so append without writing the header
        #         df7.to_csv(filename7, mode='a', index=False, header=False)
        #
        #         # Check if file exists to avoid writing headers multiple times
        #     if not os.path.isfile(filename8):
        #         df8.to_csv(filename8, index=False,
        #                    header=["Vehicle", "Time (s)", "Speed", "Compass", "Accelerometer_x", "Accelerometer_y",
        #                            "Accelerometer_z", "GNSS_latitude", "GNSS_longitude",
        #                            "GNSS_altitude", "Control_throttle", "Control_steer"])
        #     else:  # else it exists so append without writing the header
        #         df8.to_csv(filename8, mode='a', index=False, header=False)
        #
        #         # Check if file exists to avoid writing headers multiple times
        #     if not os.path.isfile(filename9):
        #         df9.to_csv(filename9, index=False,
        #                    header=["Vehicle", "Time (s)", "Speed", "Compass", "Accelerometer_x", "Accelerometer_y",
        #                            "Accelerometer_z", "GNSS_latitude", "GNSS_longitude",
        #                            "GNSS_altitude", "Control_throttle", "Control_steer"])
        #     else:  # else it exists so append without writing the header
        #         df9.to_csv(filename9, mode='a', index=False, header=False)
        #
        #         # Check if file exists to avoid writing headers multiple times
        #     if not os.path.isfile(filename10):
        #         df10.to_csv(filename10, index=False,
        #                     header=["Vehicle", "Time (s)", "Speed", "Compass", "Accelerometer_x", "Accelerometer_y",
        #                             "Accelerometer_z", "GNSS_latitude", "GNSS_longitude",
        #                             "GNSS_altitude", "Control_throttle", "Control_steer"])
        #     else:  # else it exists so append without writing the header
        #         df10.to_csv(filename10, mode='a', index=False, header=False)

        if args.sync and synchronous_master:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)

        print('\ndestroying %d vehicles' % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(all_id), 2):
            all_actors[i].stop()

        print('\ndestroying %d walkers' % len(walkers_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in all_id])

        time.sleep(0.5)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')