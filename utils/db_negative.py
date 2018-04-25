# data set preparation
import pandas as pd

sep = '\t'

class NegativeSampling:
    def __init__(self, inpath, outpath, encode):
        self.inpath = inpath
        self.outpath = outpath
        self.encode = encode

    def negative_sampling(self):
        df_sim = pd.read_table(self.inpath, sep='\t', header=None, encoding='utf-8', error_bad_lines=False,
                               names=['question', 'candidate', 'label'], skip_blank_lines=True,
                               engine='python')
        fo = open(self.outpath, 'w', encoding='utf-8')
        question_num = 0

        former_q = ''
        for i in range(len(df_sim['label'])):

            sent_question = df_sim.iloc[i, 0].strip()
            if sent_question != former_q:
                question_num += 1
                start = i
                right_found = 0
                first_found = 0
                former_q = sent_question
            if sent_question == former_q:
                origin_label = int(df_sim.iloc[i, 2])
                if origin_label == 0:
                    label = -1
                else:
                    label = 1
                    if right_found:
                        continue
                    else:
                        right_ans = df_sim.iloc[i, 1].strip()
                        right_found = 1
                        first_found = 1
                try:
                    candidate = df_sim.iloc[i, 1].strip()
                except AttributeError:
                    candidate = df_sim.iloc[i, 1]

                if right_found:
                    if first_found:
                        for j in range(i - start):
                            ques = df_sim.iloc[j + start, 0].strip()
                            cand = df_sim.iloc[j + start, 1].strip()
                            la = df_sim.iloc[j + start, 2]
                            if la != 1:
                                if j % 2 == 0:
                                    fo.write(str(ques) + '\t' + str(right_ans) + '\t' + str(cand) + '\t' + '1' + '\n')
                                else:
                                    fo.write(str(ques) + '\t' + str(cand) + '\t' + str(right_ans) + '\t' + '-1' + '\n')
                        first_found = 0

                    if label != 1:
                        if i % 2 == 0:
                            fo.write(str(sent_question) + '\t' + str(right_ans) + '\t' + str(candidate) + '\t' + '1' + '\n')
                        else:
                            fo.write(str(sent_question) + '\t' + str(candidate) + '\t' + str(right_ans) + '\t' + '-1' + '\n')

        fo.close()
        print('Successfully generated! Total question number equals %d' % question_num)


    def test_file_generation(self):
        with open(self.inpath, 'r', encoding='utf-8') as fi:
            fo = open(self.outpath, 'w', encoding='utf-8')
            qa_index = 0
            for line in fi:
                if line[1] == 'q':
                    same_query = line[line.index('\t') + 1:]
                if line[1] == 't':  # 问题对应的三元组对
                    same_triple = line[line.index('\t') + 1:]
                    sub = line[line.index('\t') + 1:line.index(' |||')].strip()  # 提取triple中的主语
                    qNSub = line[line.index(' ||| ') + 5:]  # 提取triple中的谓词和答案？
                    right_pre = qNSub[:qNSub.index(' |||')]  # 提取谓词
                    candidate = kb.get_all_predicates(sub)
                    if candidate:  # and qa_index <= 1000:
                        fo.write(same_query.replace('\n', '') + sep + same_triple.replace('\n', '') + sep +
                                 '1' + '\n')
                        qa_index += 1
                        for pre, ans in candidate.items():
                            if qa_index % 10000 == 0:
                                print('already generated [%d] lines of testing pairs' % qa_index)
                            if pre != right_pre:
                                fo.write(same_query.replace('\n', '') + sep + sub + ' ||| ' + pre + ' ||| ' + ans + sep
                                         + same_triple.replace('\n', '') + sep + '-1' + '\n')
                                qa_index += 1

                    candidate.clear()

            fo.close()
            print('Testing file generated!')



if __name__ == '__main__':
    # inpath = "../Data/nlpcc-iccpol-2016.kbqa.training-data"
    outpath = "../Data/NLPCC2017-OpenDomainQA/training&testing/dbqa-pairwise.training-data"
    inpath = "../Data/NLPCC2017-OpenDomainQA/training&testing/nlpcc-iccpol-2016.dbqa.training-data"
    nega = NegativeSampling(inpath, outpath, 'utf-8')
    nega.negative_sampling()

    # testpath = 'E:/Lucy/Dissertation/code/model/NLPCC2016KBQA-master/nlpcc-iccpol-2016.kbqa.testing-data'
    testpath = '../Data/nlpcc-iccpol-2016.kbqa.testing-data'
    test_outpath = '../Data/test-data'
    # testGeneration = NegativeSampling(testpath, test_outpath, 'utf-8')
    # testGeneration.test_file_generation()
