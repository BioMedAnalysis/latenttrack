import glob
import nibabel as nib
import torch


class TractTckDataSet(object):
    def __init__(self, tck_file: str, transform=None) -> None:

        self.tck_file = tck_file

        if tck_file.split(".")[-1] != "tck":
            raise ValueError("Only tck format is supported.")

        self.tck_obj = nib.streamlines.tck.TckFile(self.tck_file)

        self.streamlines = self._load().streamlines

        self._iter = self._generator()
        self.transform = transform

    def _load(self):
        with open(self.tck_file, "rb") as f:
            return self.tck_obj.load(f)

    def line_count(self):
        return len(self.streamlines)

    def point_count(self):
        return None

    def _generator(self):
        for each in range(self.line_count()):
            yield self.streamlines[each]

    def next_line(self):
        try:
            temp = next(self._iter)
        except StopIteration:
            return None, None
        if self.transform:
            temp = self.transform(temp)
            temp = temp.astype(float)

        return temp, temp.shape[0]

    def tolist(self):
        seq_list = []
        seq_len = []

        while True:
            temp = self.next_line()

            if temp[0] is not None:
                seq_list.append(
                    torch.cat(
                        (torch.zeros(1, 3).float(), torch.from_numpy(temp[0]).float())
                    )
                )
                seq_len.append(temp[1])
            else:
                break

        return seq_list, seq_len


class FileScanner(object):
    @staticmethod
    def scan(_dir, file_type="tck"):
        filenames = glob.glob(_dir + "*." + str(file_type))

        return {
            filename.split("/")[-1].split(".")[0]: filename for filename in filenames
        }


def tck2dataset(tck_files, transform=None):
    bundle_iter = {}
    for bundle_name, bundle_file in tck_files.items():
        bundle_iter[bundle_name] = TractTckDataSet(bundle_file, transform)

    bundle_dataset = {}
    for bundle_name, _iter in bundle_iter.items():
        seq_list = []
        seq_len = []

        while True:
            temp = _iter.next_line()
            if temp[0] is not None:
                seq_list.append(
                    torch.cat(
                        (torch.zeros(1, 3).float(), torch.from_numpy(temp[0]).float())
                    )
                )
                seq_len.append(temp[1])
            else:
                bundle_dataset[bundle_name] = seq_list
                break

    # print bundle info
    for k, v in bundle_dataset.items():
        print("{}: {} streamlines".format(k, len(v)))

    # combine streamline and shuffle
    combined_streamlines = []
    for bundle, dataset in bundle_dataset.items():
        combined_streamlines += dataset

    return combined_streamlines
