import csv

class DirSet:

    def __init__(self):
        self.dir = ""
        self.mdr = {}

def foreach_dirset(input_csv_file, dir_prefix, doFunc):
    with open(input_csv_file) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')

        for row in readCSV:

            dirset = DirSet()
            dirset.dir = dir_prefix+"/"+row[0]

            metadata_file_name = row[1]

            with open(dir_prefix+"/"+metadata_file_name) as metafile:
                records = csv.DictReader(metafile)
                mdr = []
                for record in records:
                    mdr.append(record)
                dirset.mdr = mdr[0]

            doFunc(dirset)

def load_data_interp(dir, data_source):

    obs_coord = []
    if data_source == "lidar":
        data_fname = dir+"/obs_poses_interp_transform.csv"
    elif data_source == "camera":
        data_fname = dir+"/obs_poses_camera.csv"
    else:
        print "invalid data source type"
        exit(1)
        
    with open(data_fname, 'r') as f:
        reader = csv.DictReader(f)
        # load lidar/camera transformations
        for row in reader:
            obs_coord.append(row)

    return obs_coord
