import cv2


class Localizer:

    def __init__(self, waypoints):

        self.orb = cv2.ORB_create(
            nfeatures=2000
        )

        self.bf = cv2.BFMatcher(
            cv2.NORM_HAMMING
        )

        self.refs = {}

        print("\n========== LOADING REFERENCES ==========")

        for name, path in waypoints.items():

            img = cv2.imread(
                path,
                cv2.IMREAD_GRAYSCALE
            )

            if img is None:
                print(f"[ERROR] Cannot load: {path}")
                continue

            kp, des = self.orb.detectAndCompute(
                img,
                None
            )

            if des is None:
                print(f"[ERROR] No features in: {path}")
                continue

            self.refs[name] = des

            print(
                f"[OK] {name} -> "
                f"{len(kp)} keypoints"
            )

        print("========================================\n")


    def calculate_score(self, frame_des, ref_des):

        if frame_des is None:
            return 0

        if ref_des is None:
            return 0

        if len(frame_des) < 2:
            return 0

        if len(ref_des) < 2:
            return 0

        matches = self.bf.knnMatch(
            frame_des,
            ref_des,
            k=2
        )

        good = []

        for pair in matches:

            if len(pair) != 2:
                continue

            m, n = pair

            # Strict ratio test
            if m.distance < 0.65 * n.distance:
                good.append(m)

        return len(good)


    def localize(self, frame):

        if frame is None:
            return "unknown", 0

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        kp, des = self.orb.detectAndCompute(
            gray,
            None
        )

        if des is None:
            print("[WARNING] No features detected")
            return "unknown", 0


        scores = {}

        # Compare current frame with every reference
        for name, ref_des in self.refs.items():

            score = self.calculate_score(
                des,
                ref_des
            )

            scores[name] = score


        # Sort scores
        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )


        print("\n-----------------------------")
        print("CURRENT FRAME MATCH SCORES")

        for name, score in ranked:
            print(f"{name}: {score}")

        print("-----------------------------")


        if len(ranked) == 0:
            return "unknown", 0


        best_name = ranked[0][0]
        best_score = ranked[0][1]


        if len(ranked) > 1:
            second_score = ranked[1][1]
        else:
            second_score = 0


        print(
            f"BEST: {best_name} = {best_score}"
        )

        print(
            f"SECOND: {ranked[1][0]} = {second_score}"
        )


        # VERY IMPORTANT:
        # Do not accept weak matches.

        if best_score < 15:

            print("RESULT: UNKNOWN - weak match")

            return "unknown", best_score


        # Best result must clearly beat
        # second-best result.

        if best_score < second_score * 1.4:

            print("RESULT: UNKNOWN - ambiguous")

            return "unknown", best_score


        print(
            f"RESULT: {best_name}"
        )

        return best_name, best_score