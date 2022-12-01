from tqdm import tqdm

f1 = open("dataset/1_nlpcc2013data-nlpcc2013_data/train_1.txt", "r", encoding = "utf-8")
f2 = open("dataset/1_nlpcc2013data-nlpcc2013_data/train.txt", "w", encoding = "utf-8")
#没有0标签 因此把最大的8换成0
for line in tqdm(f1):
    # print(line)
    ls = line.split("\t")
    label = ls[0]
    # print(label)
    # print(ls[1])
    if label == "8":
        f2.write("0"+ "\t" + ls[1])
    else:
        f2.write(line)