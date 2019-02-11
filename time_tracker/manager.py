class Rider:

    def __init__(self, user_id):
        self.user_id = user_id          # unique user id
        self.feature_vector_list = []   # list of features, shape = [?, 2048]
        self.ride_list = []             # list of finished rides, shape = [timestamp_start, timestamp_stop]

    def add_feature(self, feature_vector):
        self.feature_vector_list.append(feature_vector)

    def start_ride(self, timestamp):
        if self.has_finished():
            self.ride_list.append([timestamp, -1])
        else:
            self.ride_list[len(self.ride_list)-1][1] = timestamp
            self.ride_list.append([timestamp, -1])
            print("[Warning] start_ride: active ride!")

    def stop_ride(self, timestamp):
        if not self.has_finished():
            self.ride_list[len(self.ride_list)-1][1] = timestamp
        else:
            print("[Warning] stop_ride: no active ride!")

    def get_feature_list(self):
        return self.feature_vector_list

    def get_user_id(self):
        return self.user_id

    def get_ride_list(self):
        return self.ride_list

    def has_finished(self):
        if self.ride_list:                                      # check if ride_list is not empty
            if self.ride_list[len(self.ride_list)-1][1]!=-1:    # check if timestamp_stop of last ride is not -1
                return True
        return False

    def is_ghost_rider(self):           # feature_vector is empty, no reference
        if not self.feature_vector_list:
            return True
        else:
            return False


class Manager:

    def __init__(self):
        self.rider_list = []                # list of active rider, shape = [rider_id, rider]

    def is_allowed_to_ride(self, user_id):
        rider_id = self.get_rider_id(user_id)

        if rider_id != -1:
            if self.ghost_rider_on_track():
                    if self.rider_list[rider_id].is_ghost_rider():
                        return [False, "Only one active ghost rider allowed"]
                    else:
                        return [True, "True"]
            else:
                return [True, "True"]
        else:
            return [False, "Check-in needed"]

    def ghost_rider_on_track(self):
        for rider in self.rider_list:
            if rider.is_ghost_rider():
                return True
        return False

    def is_active_rider(self, user_id):
        for rider in self.rider_list:
            if rider.get_user_id() == user_id:
                return True
        return False

    def get_rider_id(self, user_id):    # rider id = index of rider in rider_list
        for rider_id, rider in enumerate(self.rider_list):
            if rider.get_user_id() == user_id:
                return rider_id
        return -1

    def add_rider(self, user_id):
        if not self.is_active_rider(user_id):
            self.rider_list.append(Rider(user_id))

    def get_active_rider(self):     # active means rider which is on track
        active_rider = []

        for rider_id, rider in enumerate(self.rider_list):
            if not rider.has_finished():
                active_rider.append(rider_id)

        return active_rider
