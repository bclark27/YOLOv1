import math
from random import gauss
import numpy as np

class KalmanFilter:
    def __init__(self, init_coord_state, proc_variance, proc_covariance, measure_variance, measure_covariance):
        self.x = np.array(init_coord_state)
        self.P = np.array([[0.1, 0],
                            [0, 0.1]])
        # state transition model is set to assume that the next
        # coordinate will look the same as the last
        self.F = np.array([[1, 0],
                            [0, 1]])

        # variance and coovariance proc noise
        self.Q = np.array([[proc_variance, proc_covariance],
                           [proc_covariance, proc_variance]])
        self.H = np.array([[1, 0],
                            [0, 1]])

        # variance and coovariance measure noise
        self.R = np.array([[measure_variance, measure_covariance],
                            [measure_covariance, measure_variance]])

    def predict_next_state(self):
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        self.P = P_pred
        self.x = x_pred
        return x_pred

    def update_kalman_state(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        x_prime = self.x + K @ y
        P_prime = self.P - K @ self.H @ self.P

        self.x = x_prime
        self.P = P_prime

    def update(self, new_coord):
        self.predict_next_state()
        self.update_kalman_state(np.array(new_coord))

    def get_state(self):
        # return list(self.x)
        return list(self.predict_next_state())

class DetectedObject:
    def __init__(self, init_box, proc_variance, proc_covariance, measure_variance, measure_covariance):
        """
        :param: init_box: List containing
            - [x1, y1]
            - [x2, y2]
            - predicted class name
            - predicted class probability
        """

        self.kalman_coord1 = KalmanFilter(init_box[0], proc_variance, proc_covariance, measure_variance, measure_covariance)
        self.kalman_coord2 = KalmanFilter(init_box[1], proc_variance, proc_covariance, measure_variance, measure_covariance)

        self.label = init_box[2]
        self.prob = init_box[3]
        self.since_last_update = 0
        self.am_new_object = True
        self.since_creation = 1

    def get_box(self):
        """
        :return: List containing:
            - [x1, y1]
            - [x2, y2]
            - predicted class name
            - predicted class probability
        """

        c1 = self.kalman_coord1.get_state()
        c2 = self.kalman_coord2.get_state()

        c1 = [int(x) for x in c1]
        c2 = [int(x) for x in c2]

        return [c1, c2, self.label, self.prob]

    def iou_score(self, new_obj):
        """
        :param new_obj: List containing:
            - (x1, y1)
            - (x2, y2)
            - predicted class name
            - predicted class probability
        """

        # if not the same class then don't even consider the obj
        if new_obj[2] != self.label:
            return 0

        c1 = new_obj[0]
        c2 = new_obj[1]

        cur_coord1 = self.kalman_coord1.get_state()
        cur_coord2 = self.kalman_coord2.get_state()

        intersection_x1 = max(c1[0], cur_coord1[0])
        intersection_y1 = max(c1[1], cur_coord1[1])
        intersection_x2 = min(c2[0], cur_coord2[0])
        intersection_y2 = min(c2[1], cur_coord2[1])

        intersection_area = max(0, (intersection_x2 - intersection_x1)) * max(0, (intersection_y2 - intersection_y1))
        my_area = abs((cur_coord2[0] - cur_coord1[0]) * (
                cur_coord2[1] - cur_coord1[1]))
        new_area = abs((c2[0] - c1[0]) * (
                c2[1] - c1[1]))

        # make sure denominator is not zero
        return intersection_area / (new_area + my_area - intersection_area + 0.000001)

    def update(self, new_obj):
        """
        :param new_obj: List containing:
            - (x1, y1)
            - (x2, y2)
            - predicted class name
            - predicted class probability
        """

        c1 = new_obj[0]
        c2 = new_obj[1]

        self.kalman_coord1.update(c1)
        self.kalman_coord2.update(c2)
        self.prob = new_obj[3]
        self.since_last_update = 0

    def inc_since_last_update(self):
        self.since_last_update += 1

    def get_since_last_update(self):
        return self.since_last_update

    def inc_since_creation(self):
        self.since_creation += 1

    def am_new_obj(self):
        #return self.am_new_object
        return False

    def get_since_creation(self):
        return self.since_creation

    def make_as_not_new(self):
        self.am_new_object = False

class ObjectDetector:
    def __init__(self, new_obj_time, obj_life, match_tolerance, proc_variance, proc_covariance, measure_variance, measure_covariance):
        self.new_obj_time = new_obj_time
        self.obj_life = obj_life
        self.match_tolerance = match_tolerance
        self.proc_variance = proc_variance
        self.proc_covariance = proc_covariance
        self.measure_variance = measure_variance
        self.measure_covariance = measure_covariance

        self.live_objects = []

    def update_objects(self, objects):
        """
        :param objects: List of lists containing:
            - (x1, y1)
            - (x2, y2)
            - predicted class name
            - predicted class probability
        """

        live_objs_used = [False] * len(self.live_objects)
        objects_used = [False] * len(objects)
        # for each new object, find the existing box which best suits it
        k = 0
        for obj in objects:
            # make sure if the live objs are exausted then break
            if False not in live_objs_used:
                break

            i = 0
            best_index = 0
            best_score = -1
            best_found = False
            for obj_detector in self.live_objects:
                # if the object has been updated then skip
                if live_objs_used[i]:
                    i += 1
                    continue

                score = obj_detector.iou_score(obj)

                # only allow the obj to take the box if it atleast has some commonality, and is the best box so far
                if score > best_score and score > self.match_tolerance:
                    best_found = True
                    best_score = score
                    best_index = i
                i += 1

            if not best_found:
                k += 1
                continue

            # update the best box and mark it as used and mark the obj used as used
            self.live_objects[best_index].update(obj)
            live_objs_used[best_index] = True
            objects_used[k] = True

            k += 1


        # init death array
        dead_objs = [False] * len(self.live_objects)
        # inc all the new objects since creation counter
        i = 0
        for obj_detector in self.live_objects:

            # inc all boxes since last updated
            obj_detector.inc_since_last_update()

            # if the object is new then inc since its creation
            if obj_detector.am_new_obj():
                obj_detector.inc_since_creation()

                # if a new obj has crossed the threshold of becoming a long lasting obj then mark it as not new
                if obj_detector.get_since_creation() > self.new_obj_time:
                    obj_detector.make_as_not_new()

            # if an object is too old and hasnt updated in too long then mark it for death
            if obj_detector.get_since_last_update() > self.obj_life:
                dead_objs[i] = True

            i += 1

        self.live_objects = [obj for (obj, remove) in zip(self.live_objects, dead_objs) if not remove]

        # if there are any new objects left that were not taken then create new detectors for them
        i = 0
        for obj in objects:
            if objects_used[i]:
              continue

            c1 = obj[0]
            c2 = obj[1]
            label = obj[2]

            new_detector = DetectedObject(obj, self.proc_variance, self.proc_covariance, self.measure_variance, self.measure_covariance)
            self.live_objects.append(new_detector)

    def get_best_boxes(self):
        best_boxes = []
        for detector in self.live_objects:
            if not detector.am_new_obj():
                best_boxes.append(detector.get_box())
        return best_boxes