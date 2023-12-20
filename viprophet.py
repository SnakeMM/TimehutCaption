import os, argparse, time, csv
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler

"""
处理输入的数据格式：
id,国家,客户端,生日,性别,创建者身份,上传总量,总容量,有上传的天数,有数据的天数,家人数量,邀请的时间,最近来访数量,点赞数,评论数,描述数,是否vip
537745430,KR,android,2014-01-01,girl,mom,35,30.67808723449707,1,2,0,0,1,0,0,0,false
"""

banlist = []

def read_banlist():
    ban_file = 'test_baby_ids.csv'
    if os.path.exists(ban_file):
        with open(ban_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                banlist.append(row[0])


def read_data(args):
    data_file = args.data_file
    data = []
    labels = []
    originCount = 0
    bannedCount = 0
    birthInvalidCount = 0
    if os.path.exists(data_file):
        with open(data_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                feature = []
                label = []
                # 去掉标题栏
                if (row[0].isdigit()):
                    originCount += 1
                    if row[0] in banlist:
                        bannedCount += 1
                        continue

                    length = len(row)

                    # 第一列是 babyId，后续列是 feature
                    birthInvalid = False
                    for i in range(2, length-1):
                        if i == 4:
                            # 取孩子被创建当时的年龄（天数）
                            birthDate = row[i]
                            if (birthDate.startswith('1')):
                                # 异常的生日 10000 年
                                birthInvalid = True
                                break
                                
                            createDate = row[1][:10]
                            days = calcDaysBetweenDates(birthDate, createDate)
                            #if days < 0:
                                # 不算出生前
                                #birthInvalid = True
                            feature.append(days)
                        else:
                            feature.append(parseElement(row[i]))

                    if birthInvalid:
                        birthInvalidCount += 1
                        continue
                    data.append(feature)
                    #print(feature)

                    # 最后一列是 vip 标志
                    label.append(parseLabel(row[length-1]))
                    labels.append(label)
                    #print(label)
    

    print('读取原始 baby 数:', originCount)
    print('读取测试 baby 数:', len(banlist))
    print('移除测试 baby 数:', bannedCount)
    print('异常生日 baby 数:', birthInvalidCount)
    print('训练有效 baby 数:', len(data))

    return data, labels

def calcDaysBetweenDates(str1, str2):
    date1 = datetime.strptime(str1, '%Y-%m-%d')
    date2 = datetime.strptime(str2, '%Y-%m-%d')
    delta = date2 - date1
    return float(delta.days)

def parseElement(element : str):
    return float(element) if element.strip().replace('.', '', 1).isdigit() else element.strip()

def parseLabel(label : str):
    return 1.0 if label.lower() == "true" else 0.0

"""
训练与推理的代码
来自万超给的 demo
"""
class PredictNet(nn.Module):
    def __init__(self, input_dim):
        super(PredictNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
    
def data_to_tensor(data, encoding_info=None):
    df = pd.DataFrame(data)
    # 独热编码
    str_cols = df.select_dtypes(include=['object', 'string']).columns
    df_encoded = pd.get_dummies(df, columns=str_cols)
    # 如果传入encoding_info说明为测试阶段补充缺失的独热列
    if encoding_info:
        for col, encoded_cols in encoding_info.items():
            for encoded_col in encoded_cols:
                if encoded_col not in df_encoded.columns:
                    df_encoded[encoded_col] = 0
    
    # 确保所有列名都是字符串类型
    df_encoded.columns = df_encoded.columns.astype(str)
    # 归一化
    num_cols = df_encoded.select_dtypes(include=['int64', 'float64']).columns
    scaler = MinMaxScaler()
    df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])
    df_encoded = df_encoded.astype(float) # 全部转为 Float
    # 转换回 Tensor
    data_tensor = torch.tensor(df_encoded.values).float()
    # 记录编码信息，用于处理训练数据one-hot后和推理数据one-hot维度不一致
    str_cols_dict = {col: [str(encoded_col) for encoded_col in df_encoded.columns if str(encoded_col).startswith(str(col) + '_')] for col in str_cols}
    return data_tensor, str_cols_dict

def process_data(data, labels, training_ratio, validation_ratio, test_ratio, batch_size=64):
    if (training_ratio + validation_ratio + test_ratio) > 1:
        raise Exception(f'\n比例不能总和不能超过1，建议0.6:0.2:0.2')
    
    data_tensor, str_cols_dict = data_to_tensor(data)
    labels_tensor = torch.tensor(labels).float()
    # 将数据和标签包装成 TensorDataset
    dataset = TensorDataset(data_tensor, labels_tensor)

    # 定义训练集和测试集的大小
    train_size      = int(training_ratio * len(dataset))
    validation_size = int(validation_ratio * len(dataset))
    test_size       = len(dataset) - train_size - validation_size

    # 随机分割数据集
    train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])

    train_loader      = DataLoader(train_dataset,      batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader       = DataLoader(test_dataset,       batch_size=batch_size, shuffle=False)
    return train_loader, validation_loader, test_loader, str_cols_dict

def eval(model, data_loader, loss_func):
    model.eval()
    total_val_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    # 获取一批数据
    features, labels = next(iter(data_loader))
    # 分析标签
    unique_labels = torch.unique(labels)
    num_unique_labels = len(unique_labels)
    
    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            outputs = model(inputs)
            
            # 计算损失
            val_loss = loss_func(outputs, labels)
            total_val_loss += val_loss.item()
            
            # 计算准确度
            if num_unique_labels <= 2:
                predicted = outputs >= 0.5
            else:
                # 理论上不会进这里了
                _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
            
        avg_val_loss = total_val_loss / len(data_loader)
        accuracy = correct_predictions / total_predictions
        return avg_val_loss, accuracy

def train(train_loader, validation_loader, epoch_times):
    print('开始训练')
    # 计算输入数据维度
    data_iter = iter(train_loader)
    batch = next(data_iter)
    input_dim = batch[0].shape[1]
    
    model = PredictNet(input_dim) # 实例化模型
    loss_func = nn.BCELoss() # 损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001) # 优化器
    
    best_val_loss = float('inf')
    best_val_accuracy = None
    best_model_state_dict = None
    # 训练模型
    average_loss = 0.0
    total_loss = 0.0
    total_batches = 0
    for epoch in tqdm(range(epoch_times)):  # 迭代次数
        model.train()
        for data in train_loader:
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1
            
        if epoch % 10 == 0:
            # 去验证集测试一下，最终只保留最好的模型
            avg_val_loss, val_accuracy = eval(model, validation_loader, loss_func)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_accuracy = val_accuracy
                best_model_state_dict = model.state_dict()
            average_loss = total_loss / total_batches
            tqdm.write(f'Epoch {epoch+1}, Average Loss: {average_loss:.4f}, Accuracy: {f"{val_accuracy * 100:.2f}".rstrip("0").rstrip(".")}%')
    
    # 还原最佳状态    
    model.load_state_dict(best_model_state_dict)
    return model, average_loss, best_val_accuracy, loss_func

def save_model(model, output_dir, str_cols_dict, accuracy=None):
    model_name = f'predict_{time.strftime("%y%m%d%H%M%S", time.localtime(time.time()))}{("_" + "{:.5f}".format(accuracy).rstrip("0").rstrip(".")) if accuracy else ""}.pth'
    output_path = os.path.join(output_dir if output_dir else os.getcwd(), model_name)
    
    # 将模型状态和编码信息打包到一个字典中
    saved_content = {
        # "model_state": model.state_dict(),
        "model": model,
        "encoding_info": str_cols_dict
    }
    torch.save(saved_content, output_path)
    print('模型保存在:', output_path)

def main(args):
    print('准备训练预测模型')

    read_banlist()
    data, labels = read_data(args)
    train_loader, validation_loader, test_loader, str_cols_dict = process_data(data, labels, args.training_ratio, args.validation_ratio, args.test_ratio, args.batch_size)
    best_model, loss, best_val_accuracy, loss_func = train(train_loader, validation_loader, args.epoch)
    _, test_accuracy = eval(best_model, test_loader, loss_func)
    _, validation_accuracy = eval(best_model, validation_loader, loss_func)
    print(test_accuracy)
    print(f'训练完成，共训练 {args.epoch} 轮，训练Loss: {loss:.4f}, 测试集准确率：{f"{(test_accuracy * 100):.2f}".rstrip("0").rstrip(".")}%, 验证集准确率：{f"{(validation_accuracy * 100):.2f}".rstrip("0").rstrip(".")}%')
    
    # 保存模型
    if args.output_dir:
        save_model(best_model, args.output_dir, str_cols_dict, accuracy=test_accuracy)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VIP购买预测模型训练推理脚本")

    # 训练
    parser.add_argument("--data_file", type=str, default='', help="数据源csv")
    parser.add_argument("--output_dir", type=str, default='', help="保存模型的目录")
    parser.add_argument("--training_ratio", type=float, default=0.6, help="训练集比例")
    parser.add_argument("--validation_ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="测试集比例")
    parser.add_argument("--batch_size", type=int, default=64, help="批训练数据")
    parser.add_argument("--epoch", type=int, default=100, help="迭代次数")
    
    # 推理
    parser.add_argument("--predict_model", type=str, default='', help="推理模型路径")
    parser.add_argument("--predict_data", type=str, default='', help="预测数据")
    
    args = parser.parse_args()
    main(args)