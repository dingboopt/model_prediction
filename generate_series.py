from utils import generate_exp_series
import sys
if __name__ == "__main__":
    result = generate_exp_series(int(sys.argv[2]), int(sys.argv[1]), reverse = sys.argv[3]=='1')
    output = [str(x) for x in result]
    print(" ".join(output))