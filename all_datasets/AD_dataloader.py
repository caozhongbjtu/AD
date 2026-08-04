import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')



class PSMSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, 'train.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data#[:(int)(len(data) * 0.8)]
        self.val = data[(int)(len(data) * 0.8):]
        self.test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]
        print("train:", self.train.shape)
        print("val:", self.val.shape)
        print("test:", self.test.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 2

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]).T, np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]).T, np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]).T, np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            start = index // self.step * self.win_size
            if start + self.win_size > self.test.shape[0]:
                start = self.test.shape[0] - self.win_size
            return np.float32(self.test[start:start + self.win_size]).T, \
                   np.float32(self.test_labels[start:start + self.win_size])
            # return np.float32(self.test[
            #                   index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]).T, np.float32(
            #     self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        # discrete_channels = range(1, 55)
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        # data = np.delete(data, discrete_channels, axis=-1)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        # test_data = np.delete(test_data, discrete_channels, axis=-1)
        self.test = self.scaler.transform(test_data)
        self.train = data#[:(int)(len(data) * 0.8)]
        self.val = data[(int)(len(data) * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
        print("train:", self.train.shape)
        print("val:", self.val.shape)
        print("test:", self.test.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 2

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]).T, np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]).T, np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]).T, np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            start = index // self.step * self.win_size
            if start + self.win_size > self.test.shape[0]:
                start = self.test.shape[0] - self.win_size
            return np.float32(self.test[start:start + self.win_size]).T, \
                   np.float32(self.test_labels[start:start + self.win_size])
            # return np.float32(self.test[
            #                   index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]).T, np.float32(
            #     self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        discrete_channels = range(1, 25)
        
        
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        data = np.delete(data, discrete_channels, axis=-1)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))

        test_data = np.delete(test_data, discrete_channels, axis=-1)

        self.test = self.scaler.transform(test_data)
        self.train = data[:(int)(len(data) * 0.8)]
        data_len = len(self.train)
        self.val = data[(int)(len(data) * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))
        print("train:", self.train.shape)
        print("val:", self.val.shape)
        print("test:", self.test.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 2

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]).T, np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]).T, np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]).T, np.float32(self.test_labels[index:index + self.win_size])
           
            
        else:
            start = index // self.step * self.win_size
            if start + self.win_size > self.test.shape[0]:
                start = self.test.shape[0] - self.win_size
            return np.float32(self.test[start:start + self.win_size]).T, \
                   np.float32(self.test_labels[start:start + self.win_size])
            # return np.float32(self.test[
            #                   index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]).T, np.float32(
            #     self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data[:(int)(len(data) * 0.8)]
        data_len = len(self.train)
        self.val = data[(int)(len(data) * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))
        print("train:", self.train.shape)
        print("val:", self.val.shape)
        print("test:", self.test.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 2

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]).T, np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]).T, np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]).T, np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            start = index // self.step * self.win_size
            if start + self.win_size > self.test.shape[0]:
                start = self.test.shape[0] - self.win_size
            return np.float32(self.test[start:start + self.win_size]).T, \
                   np.float32(self.test_labels[start:start + self.win_size])
            # return np.float32(self.test[
            #                   index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]).T, np.float32(
            #     self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SWATSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data[:(int)(len(train_data) * 0.8)]
        self.test = test_data
        self.val = train_data[(int)(len(train_data) * 0.8):]
        self.test_labels = labels
        print("train:", self.train.shape)
        print("val:", self.val.shape)
        print("test:", self.test.shape)
        
    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 2

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]).T, np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]).T, np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]).T, np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            start = index // self.step * self.win_size
            if start + self.win_size > self.test.shape[0]:
                start = self.test.shape[0] - self.win_size
            return np.float32(self.test[start:start + self.win_size]).T, \
                   np.float32(self.test_labels[start:start + self.win_size])
            # return np.float32(self.test[
            #                   index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]).T, np.float32(
            #     self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class TSADSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        # ======================
        # 1. 读取 + 重构（保留你原来的逻辑）
        # ======================
        data = self._read_data(root_path)

        labels = data["label"].values.reshape(-1, 1)
        data = data.drop(columns=["label"]).values

        # ======================
        # 2. 简单切分（和 SWAT 一样）CalIt2 2520
        # ======================
        if 'GECCO' in root_path:
            split = 69260
        elif 'CICIDS' in root_path:
            split = 85115
        elif 'DLR' in root_path:
            split = 11565
        elif 'NYC' in root_path:
            split = 13104 
        elif 'Creditcard' in root_path:
            split = 142403
        else:
            split = 60000
        train_data = data[:split]
        test_data = data[split:]
        test_labels = labels[split:]

        # ======================
        # 3. 标准化
        # ======================
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)

        self.train = train_data[:int(split * 0.8)]
        self.val = train_data[int(split * 0.8):]
        self.test = test_data
        self.test_labels = test_labels

        print("train:", self.train.shape)
        print("val:", self.val.shape)
        print("test:", self.test.shape)

    # ======================
    # ⭐ 核心：原始数据重构逻辑（保留）
    # ======================
    def _read_data(self, root_path):
        if 'GECCO' in root_path:
            data = pd.read_csv(os.path.join(root_path, 'GECCO.csv'))
        elif 'CICIDS' in root_path:
            data = pd.read_csv(os.path.join(root_path, 'CICIDS.csv'))
        elif 'DLR' in root_path:
            data = pd.read_csv(os.path.join(root_path, 'DLR.csv'))
        elif 'NYC' in root_path:
            data = pd.read_csv(os.path.join(root_path, 'NYC.csv'))
        elif 'Creditcard' in root_path:
            data = pd.read_csv(os.path.join(root_path, 'Creditcard.csv'))    
        else:
            data = pd.read_csv(os.path.join(root_path, 'SWAN.csv'))

        cols_name = data["cols"].unique()
        all_points = data.shape[0]

        # 判断每个变量长度
        if data.columns[0] == "date":
            n_points = data.iloc[:, 2].value_counts().max()
        else:
            n_points = data.iloc[:, 1].value_counts().max()

        n_cols = all_points // n_points

        df = pd.DataFrame()

        # ===== 多变量情况 =====
        if data.columns[0] == "date":
            df["date"] = data.iloc[:n_points, 0]
            col_data = {
                cols_name[j]: data.iloc[j * n_points:(j + 1) * n_points, 1].tolist()
                for j in range(n_cols)
            }
            df = pd.concat([df, pd.DataFrame(col_data)], axis=1)
            df.set_index("date", inplace=True)
        else:
            col_data = {
                cols_name[j]: data.iloc[j * n_points:(j + 1) * n_points, 0].tolist()
                for j in range(n_cols)
            }
            df = pd.concat([df, pd.DataFrame(col_data)], axis=1)

        # label 处理
        last_col = df.columns[-1]
        df.rename(columns={last_col: "label"}, inplace=True)

        return df

    # ======================
    # 长度
    # ======================
    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            # return int(np.ceil(self.test.shape[0] / self.win_size))
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    # ======================
    # 取数据（完全 SWAT 风格）
    # ======================
    def __getitem__(self, index):
        index = index * self.step

        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]).T, \
                   np.float32(self.test_labels[:self.win_size])

        elif self.flag == "val":
            return np.float32(self.val[index:index + self.win_size]).T, \
                   np.float32(self.test_labels[:self.win_size])

        elif self.flag == "test":
            return np.float32(self.test[index:index + self.win_size]).T, \
                   np.float32(self.test_labels[index:index + self.win_size])

        else:
            start = index // self.step * self.win_size
            if start + self.win_size > self.test.shape[0]:
                start = self.test.shape[0] - self.win_size
            return np.float32(self.test[start:start + self.win_size]).T, \
                   np.float32(self.test_labels[start:start + self.win_size])
