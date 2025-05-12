import sys
def feature_exaction():
    lines = sys.stdin.read().splitlines()
    if not lines:
        print(-1)
        return
    try:
        # 第一行为文档数量N
        n = int(lines[0])
    except:
        print(-1)
        return

    # 检查是否有足够的行：N行文档 + 1 行k
    if len(lines) != n + 2:
        print(-1)
        return
    try:
        m = int(lines[-1])
    except:
        print(-1)
        return
    
    # 处理数据
    positive_all = 0
    negative_all = 0
    word_dict = {}
    for i in range(1, n+1):
        doc = lines[i]
        if '\t' in doc:
            label, text = doc.split('\t')
            text = text.split()
        else:
            temp = doc.split()
            label = temp[0]
            text = temp[1:]
        if label == "positive":
            positive_all += 1
        else:
            negative_all += 1
        words = set(text)
        
        for word in words:
            if word not in word_dict:
                word_dict[word] = [0, 0] # 初始正负样本count       
            if label == "positive":
                word_dict[word][1] += 1
            else:
                word_dict[word][0] += 1 
    features_x2 = []
    for word, counts in word_dict.items():
        x_2 = 0.0
        TP = counts[1]
        TN = counts[0]
        FP = positive_all - counts[1]
        FN = negative_all - counts[0]
        T = TP + TN
        F = FP + FN
        if T > 0 and positive_all > 0: # word 出现在positive
            E = (T * positive_all)/n
            x_2 += (TP - E) ** 2/E
        if T > 0 and negative_all > 0: # word 没有出现在positive
            E = (T * negative_all)/n
            x_2 += (TN - E) ** 2/E
        if F > 0 and positive_all > 0: # word 出现在negative
            E = (F * positive_all)/n
            x_2 += (FP - E) ** 2/E
        if F > 0 and negative_all > 0:  # word 没有出现在negative
            E = (F * negative_all)/n
            x_2 += (FN - E) ** 2/E
        features_x2.append((word, x_2))
    features_x2.sort(key=lambda item: [-item[1], item[0]])
    for i in range(min(m, len(features_x2))):
        print(features_x2[i][0])

if __name__ == "__main__":
    feature_exaction()