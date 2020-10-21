import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create heatmap of the heard with sample overlay.')
    parser.add_argument('output',
                        help='Directory path of heatmap output. (Directory will be created if does not exist)')
    parser.add_argument('activity_dir', help='Parent directory of the activity data.')
    parser.add_argument('dataset_file', help='Path of the datase file containing the samples.')
    args = parser.parse_args()

    print("Argument values:")
    print(args.output)
    print(args.activity_dir)
    print(args.dataset_file)




