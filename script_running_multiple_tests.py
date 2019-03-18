from evaluation import robustness_to_missing_parts, robustness_to_noise


def main():
    print('robustness_to_missing_parts')
    robustness_to_missing_parts.main()

    print('robustness_to_noise')
    robustness_to_noise.main()


if __name__ == "__main__":
    main()