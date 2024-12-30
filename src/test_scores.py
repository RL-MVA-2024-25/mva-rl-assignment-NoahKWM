def test_scores():
    with open("score.txt") as file:
        lines = [float(line.rstrip()) for line in file]
    one_env_performance = lines[0]
    dr_env_performance = lines[1]

    try:
        assert one_env_performance >= 3432807.680391572
        print("Test One Env Threshold 3432807: PASS")
    except AssertionError:
        print(f"Test One Env Threshold 3432807: FAIL (Score: {one_env_performance})")

    try:
        assert one_env_performance >= 1e8
        print("Test One Env Threshold 1e8: PASS")
    except AssertionError:
        print(f"Test One Env Threshold 1e8: FAIL (Score: {one_env_performance})")

    try:
        assert one_env_performance >= 1e9
        print("Test One Env Threshold 1e9: PASS")
    except AssertionError:
        print(f"Test One Env Threshold 1e9: FAIL (Score: {one_env_performance})")

    try:
        assert one_env_performance >= 1e10
        print("Test One Env Threshold 1e10: PASS")
    except AssertionError:
        print(f"Test One Env Threshold 1e10: FAIL (Score: {one_env_performance})")

    try:
        assert one_env_performance >= 2e10
        print("Test One Env Threshold 2e10: PASS")
    except AssertionError:
        print(f"Test One Env Threshold 2e10: FAIL (Score: {one_env_performance})")

    try:
        assert one_env_performance >= 5e10
        print("Test One Env Threshold 5e10: PASS")
    except AssertionError:
        print(f"Test One Env Threshold 5e10: FAIL (Score: {one_env_performance})")

    try:
        assert dr_env_performance >= 1e10
        print("Test DR Env Threshold 1e10: PASS")
    except AssertionError:
        print(f"Test DR Env Threshold 1e10: FAIL (Score: {dr_env_performance})")

    try:
        assert dr_env_performance >= 2e10
        print("Test DR Env Threshold 2e10: PASS")
    except AssertionError:
        print(f"Test DR Env Threshold 2e10: FAIL (Score: {dr_env_performance})")
    
    try:
        assert dr_env_performance >= 5e10
        print("Test DR Env Threshold 5e10: PASS")
    except AssertionError:
        print(f"Test DR Env Threshold 5e10: FAIL (Score: {dr_env_performance})")
    

if __name__ == "__main__":
    test_scores()
