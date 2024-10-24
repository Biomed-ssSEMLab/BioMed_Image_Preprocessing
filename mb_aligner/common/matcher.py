import numpy as np
from . import ransac
from rh_renderer import models
import time

class FeaturesMatcher(object):

    def __init__(self, matcher_init_fn, **kwargs):
        self._matcher = matcher_init_fn()

        self._params = {}
        # get default values if no value is present in kwargs
        #self._params["num_filtered_percent"] = kwargs.get("num_filtered_percent", 0.25)
        #self._params["filter_rate_cutoff"] = kwargs.get("filter_rate_cutoff", 0.25)
        self._params["ROD_cutoff"] = kwargs.get("ROD_cutoff", 0.92)
        self._params["min_features_num"] = kwargs.get("min_features_num", 40)

        # Parameters for the RANSAC
        self._params["model_index"] = kwargs.get("model_index", 3)
        self._params["iterations"] = kwargs.get("iterations", 5000)
        self._params["max_epsilon"] = kwargs.get("max_epsilon", 30.0)
        self._params["min_inlier_ratio"] = kwargs.get("min_inlier_ratio", 0.01)
        self._params["min_num_inlier"] = kwargs.get("min_num_inlier", 7)
        self._params["max_trust"] = kwargs.get("max_trust", 3)
        self._params["det_delta"] = kwargs.get("det_delta", None)
        self._params["max_stretch"] = kwargs.get("max_stretch", None)
        self._params["avoid_robust_filter"] = kwargs.get("avoid_robust_filter", False)
        self._params["max_rot_deg"] = kwargs.get("max_rot_deg", None)

        self._params["use_regularizer"] = kwargs.get("use_regularizer", False)
        self._params["regularizer_lambda"] = kwargs.get("regularizer_lambda", 0.1)
        self._params["regularizer_model_index"] = kwargs.get("regularizer_model_index", 1)

        self._params["best_k_matches"] = kwargs.get("best_k_matches", 0) 

        self._params["max_distance"] = kwargs.get("max_distance", None)

    def match(self, features_kps1, features_descs1, features_kps2, features_descs2, sec1_layer, sec2_layer, load):
        if features_descs1 is None or len(features_descs1) < self._params["min_features_num"] or \
           features_descs2 is None or len(features_descs2) < self._params["min_features_num"]:
            return None

        t0 = time.time()
        match_exists = False
        step = 10000
        if sec1_layer is not None and sec2_layer is not None and load is not None:
            match_exists, match_result = load.load_prev_results('pre_matches', 'features_S{}-S{}_match'.format(sec1_layer, sec2_layer))
        if match_exists:
            matches = match_result
        else:            
            matches =[[] for i in range(len(features_descs1))]
            n = 0
            descs1_group = [features_descs1[i: i + step] for i in range(0, len(features_descs1), step)]
            descs2_group = [features_descs2[i: i + step] for i in range(0, len(features_descs2), step)]
            for ind1, group1 in enumerate(descs1_group):
                for ind2, group2 in enumerate(descs2_group):
                    raw_match = self._matcher.knnMatch(group1, group2, k=2)
                    if len(raw_match[0]) == 1:  # if group2 only have one element
                        for match1 in raw_match:
                            match1[0].queryIdx += ind1 * step
                            match1[0].trainIdx += ind2 * step
                            matches[n + ind1 * step].append(match1[0])
                            n += 1
                    else:
                        for match1, match2 in raw_match:
                            match1.queryIdx += ind1 * step
                            match1.trainIdx += ind2 * step
                            match2.queryIdx += ind1 * step
                            match2.trainIdx += ind2 * step
                            matches[n + ind1 * step].append(match1)
                            matches[n + ind1 * step].append(match2)
                            n += 1
                    n = 0

            if sec1_layer is not None and sec2_layer is not None and load is not None:
                matches_pickle = [[] for i in range(len(matches))]
                for i in range(len(matches)):
                    for j in range(len(matches[i])):
                        tmp_dict = {"distance": matches[i][j].distance, "queryIdx": matches[i][j].queryIdx, "trainIdx": matches[i][j].trainIdx}
                        matches_pickle[i].append(tmp_dict)
                load.store_result('pre_matches', 'features_S{}-S{}_match'.format(sec1_layer, sec2_layer), matches_pickle)

        t1 = time.time()
        # print(f'knnMatch time: {t1 - t0:.2f}s')

        good_matches = []
        if not match_exists:  # 若不是读取'features_S{}-S{}_match.pkl'文件，则match每一项的type是cv2.DMatch
            for match in matches:
                if len(matches) > step:
                    match = sorted(match, key=lambda x: x.distance)
                if match[0].distance < self._params["ROD_cutoff"] * match[1].distance:
                    good_matches.append(match[0])
        else:                 # 若是读取'features_S{}-S{}_match.pkl'文件，则match每一项的type是dict
            for match in matches:
                if len(matches) > step:
                    match = sorted(match, key=lambda x: x["distance"])
                if match[0]["distance"] < self._params["ROD_cutoff"] * match[1]["distance"]:
                    good_matches.append(match[0])

        # 这段逻辑是: 不仅将最近点距离<ROD_cutoff*次近点距离作为good_matches的判断标准，也将所有匹配点中距离最近的前N个匹配点作为判断标准保留
        # # least_dists = []
        # for i in range(len(matches)):
        #     # match = matches[i]
        #     matches[i] = sorted(matches[i], key=lambda x: x.distance)
        #     least_dists.append(matches[i][0].distance)

        # good_matches_idx = np.argpartition(least_dists, 50000)[:50000]
        # for i in range(50000):
        #     good_matches.append(matches[good_matches_idx[i]][0])

        # for i in range(len(matches)):
        #     match = matches[i]
        #     if match[0].distance < self._params["ROD_cutoff"] * match[1].distance and i not in good_matches_idx:
        #         good_matches.append(match[0])

        # print("good_matches length: {}".format(len(good_matches)))

        if not match_exists:
            match_points = (features_kps1[[m.queryIdx for m in good_matches]],
                            features_kps2[[m.trainIdx for m in good_matches]],
                            np.array([m.distance for m in good_matches]))
        else:
            match_points = (features_kps1[[m["queryIdx"] for m in good_matches]],
                            features_kps2[[m["trainIdx"] for m in good_matches]],
                            np.array([m["distance"] for m in good_matches]))
        return match_points

    def match_and_filter(self, features_kps1, features_descs1, features_kps2, features_descs2, sec1_layer=None, sec2_layer=None, load=None):
        match_points = self.match(features_kps1, features_descs1, features_kps2, features_descs2, sec1_layer, sec2_layer, load)

        if match_points is None:
            print("match_points is none")
            return None, None
        
       # print("the model_index in matcher.py is: ", self._params['model_index'])

        model, filtered_matches, mask = ransac.filter_matches(match_points, match_points, self._params['model_index'],
                    self._params['iterations'], self._params['max_epsilon'], self._params['min_inlier_ratio'],
                    self._params['min_num_inlier'], self._params['max_trust'], self._params['det_delta'], self._params['max_stretch'],
                    self._params['max_rot_deg'], robust_filter=not self._params['avoid_robust_filter'], max_distance=self._params['max_distance'])

        if model is None:
            print("model is none")
            return None, None

        if self._params["use_regularizer"]:
            regularizer_model, _, _ = ransac.filter_matches(match_points, match_points, self._params['regularizer_model_index'],
                        self._params['iterations'], self._params['max_epsilon'], self._params['min_inlier_ratio'],
                        self._params['min_num_inlier'], self._params['max_trust'], self._params['det_delta'], self._params['max_stretch'],
                        self._params['max_rot_deg'], robust_filter=not self._params['avoid_robust_filter'], max_distance=self._params['max_distance'])

            if regularizer_model is None:
                print("model after filter is none")
                return None, None

            result = model.get_matrix() * (1 - self._params["regularizer_lambda"]) + regularizer_model.get_matrix() * self._params["regularizer_lambda"]
            model = models.AffineModel(result)

        if self._params['best_k_matches'] > 0 and self._params['best_k_matches'] < len(filtered_matches[0]):
            # Only keep the best K matches out of the filtered matches
            # best_k_matches_idxs = np.argpartition(match_points[2][mask], -self._params['best_k_matches'])[-self._params['best_k_matches']:]  # 代码写错了，本意是保存dist最小的前K个，结果保存了最大的
            best_k_matches_idxs = np.argpartition(match_points[2][mask], self._params['best_k_matches'])[:self._params['best_k_matches']]
            filtered_matches = np.array([match_points[0][mask][best_k_matches_idxs], match_points[1][mask][best_k_matches_idxs]])

        return model, filtered_matches
