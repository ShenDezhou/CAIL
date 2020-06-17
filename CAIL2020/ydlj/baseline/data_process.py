from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import gzip
import pickle
from tqdm import tqdm
from transformers import BertTokenizer




class Example(object):

    def __init__(self,
                 qas_id,
                 qas_type,
                 doc_tokens,
                 question_text,
                 sent_num,
                 sent_names,
                 sup_fact_id,
                 para_start_end_position,
                 sent_start_end_position,
                 entity_start_end_position,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None):
        self.qas_id = qas_id
        self.qas_type = qas_type
        self.doc_tokens = doc_tokens
        self.question_text = question_text
        self.sent_num = sent_num
        self.sent_names = sent_names
        self.sup_fact_id = sup_fact_id
        self.para_start_end_position = para_start_end_position
        self.sent_start_end_position = sent_start_end_position
        self.entity_start_end_position = entity_start_end_position
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 qas_id,
                 doc_tokens,
                 doc_input_ids,
                 doc_input_mask,
                 doc_segment_ids,
                 query_tokens,
                 query_input_ids,
                 query_input_mask,
                 query_segment_ids,
                 sent_spans,
                 sup_fact_ids,
                 ans_type,
                 token_to_orig_map,
                 start_position=None,
                 end_position=None):

        self.qas_id = qas_id
        self.doc_tokens = doc_tokens
        self.doc_input_ids = doc_input_ids
        self.doc_input_mask = doc_input_mask
        self.doc_segment_ids = doc_segment_ids

        self.query_tokens = query_tokens
        self.query_input_ids = query_input_ids
        self.query_input_mask = query_input_mask
        self.query_segment_ids = query_segment_ids

        self.sent_spans = sent_spans
        self.sup_fact_ids = sup_fact_ids
        self.ans_type = ans_type
        self.token_to_orig_map=token_to_orig_map

        self.start_position = start_position
        self.end_position = end_position




def check_in_full_paras(answer, paras):
    full_doc = ""
    for p in paras:
        full_doc += " ".join(p[1])
    return answer in full_doc


def read_examples( full_file):

    with open(full_file, 'r', encoding='utf-8') as reader:
        full_data = json.load(reader)    # 完整的原始数据


    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    cnt = 0
    examples = []
    for case in tqdm(full_data):   # 遍历每个样本
        key = case['_id']
        qas_type = "" #case['type']
        sup_facts = set([(sp[0], sp[1]) for sp in case['supporting_facts']])   # TODO: 为啥是个集合？为了去重？
        sup_titles = set([sp[0] for sp in case['supporting_facts']])  # sup para 的title 列表
        orig_answer_text = case['answer']

        sent_id = 0
        doc_tokens = []
        sent_names = []
        sup_facts_sent_id = []
        sent_start_end_position = []
        para_start_end_position = []
        entity_start_end_position = []
        ans_start_position, ans_end_position = [], []

        JUDGE_FLAG = orig_answer_text == 'yes' or orig_answer_text == 'no' or orig_answer_text=='unknown'  or orig_answer_text=="" # judge_flag??
        FIND_FLAG = False

        char_to_word_offset = []  # Accumulated along all sentences
        prev_is_whitespace = True

        # for debug
        titles = set()
        para_data=case['context']
        for paragraph in para_data:  # 选中的段落
            title = paragraph[0]
            sents = paragraph[1]   # 句子列表


            titles.add(title)  # 选中的title
            is_gold_para = 1 if title in sup_titles else 0  # 是否是gold para

            para_start_position = len(doc_tokens)  # 刚开始doc_tokens是空的

            for local_sent_id, sent in enumerate(sents):  # 处理段落的每个句子
                if local_sent_id >= 100:  # 句子数量限制：一个段落最多只允许10个句子
                    break

                # Determine the global sent id for supporting facts
                local_sent_name = (title, local_sent_id)   # （title， 句子在段落中的位置）
                sent_names.append(local_sent_name)  # 作为句子的名字
                if local_sent_name in sup_facts:
                    sup_facts_sent_id.append(sent_id)   # TODO： 这个跟原始的sup标签有啥区别
                sent_id += 1   # 这个句子的id是以整个article为范围的，为什么?
                sent=" ".join(sent)
                sent += " "

                sent_start_word_id = len(doc_tokens)           # 句子开始位置的word id
                sent_start_char_id = len(char_to_word_offset)  # 句子开始位置的char id

                for c in sent:   # 遍历整个句子的字符，简历char到word之间的映射关系
                    if is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                    char_to_word_offset.append(len(doc_tokens) - 1)

                sent_end_word_id = len(doc_tokens) - 1  # 句子结尾的word位置
                sent_start_end_position.append((sent_start_word_id, sent_end_word_id))  # 句子开始和结束的位置，以元祖形式保存

                # Answer char position
                answer_offsets = []
                offset = -1

                tmp_answer=" ".join(orig_answer_text)
                while True:

                    offset = sent.find(tmp_answer, offset + 1)
                    if offset != -1:
                        answer_offsets.append(offset)   # 把所有相同答案的开始位置都找到
                    else:
                        break

                # answer_offsets = [m.start() for m in re.finditer(orig_answer_text, sent)]
                if not JUDGE_FLAG and not FIND_FLAG and len(answer_offsets) > 0:
                    FIND_FLAG = True   # 标志找到了答案，TODO：这个有啥用
                    for answer_offset in answer_offsets:
                        start_char_position = sent_start_char_id + answer_offset   # 答案开始的char位置
                        end_char_position = start_char_position + len(tmp_answer) - 1  # 答案结束的char位置
                        # 答案开始的token位置，每个答案都保存
                        ans_start_position.append(char_to_word_offset[start_char_position])
                        ans_end_position.append(char_to_word_offset[end_char_position])



                # Truncate longer document
                if len(doc_tokens) > 382:   # 如果大于382个词则break
                    # 这个截断会让每个段落至少有一个句子被加入，即使整个样本已经超过382，这样后面匹配entity还能匹配上吗？
                    break
            para_end_position = len(doc_tokens) - 1
            # 一个段落的开始和结束token位置（白空格分词）
            para_start_end_position.append((para_start_position, para_end_position, title, is_gold_para))  # 顺便加上开始和结束位置

        if len(ans_end_position) > 1:
            cnt += 1    # 如果答案结束的位置大于1，cnt+1，如果答案结束位置是0呢？
        if key <10:
            print("qid {}".format(key))
            print("qas type {}".format(qas_type))
            print("doc tokens {}".format(doc_tokens))
            print("question {}".format(case['question']))
            print("sent num {}".format(sent_id+1))
            print("sup face id {}".format(sup_facts_sent_id))
            print("para_start_end_position {}".format(para_start_end_position))
            print("sent_start_end_position {}".format(sent_start_end_position))
            print("entity_start_end_position {}".format(entity_start_end_position))
            print("orig_answer_text {}".format(orig_answer_text))
            print("ans_start_position {}".format(ans_start_position))
            print("ans_end_position {}".format(ans_end_position))
        # 一个paragraph是一个example
        example = Example(
            qas_id=key,
            qas_type=qas_type,
            doc_tokens=doc_tokens,
            question_text=case['question'],
            sent_num=sent_id + 1,
            sent_names=sent_names,
            sup_fact_id=sup_facts_sent_id,
            para_start_end_position=para_start_end_position,  # 一个样本是一个article, 有多个段落开始和结束的位置
            sent_start_end_position=sent_start_end_position,
            entity_start_end_position=entity_start_end_position,
            orig_answer_text=orig_answer_text,
            start_position=ans_start_position,   # 这里是word的开始和结束位置
            end_position=ans_end_position)
        examples.append(example)
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length, max_query_length):
    # max_query_length = 50

    features = []
    failed = 0
    for (example_index, example) in enumerate(tqdm(examples)):  # 遍历所有的example
        if example.orig_answer_text == 'yes':
            ans_type = 1
        elif example.orig_answer_text == 'no':
            ans_type = 2
        elif example.orig_answer_text == 'unknown':
            ans_type = 3
        else:
            ans_type = 0   # 统计answer type

        query_tokens = ["[CLS]"]
        for token in example.question_text.split(' '):
            query_tokens.extend(tokenizer.tokenize(token))
        if len(query_tokens) > max_query_length - 1:
            query_tokens = query_tokens[:max_query_length - 1]
        query_tokens.append("[SEP]")

        # para_spans = []
        # entity_spans = []
        sentence_spans = []
        all_doc_tokens = []
        orig_to_tok_index = []
        orig_to_tok_back_index = []
        tok_to_orig_index = [0] * len(query_tokens)

        all_doc_tokens = ["[CLS]"]   # 这一段不是啰嗦的代码吗
        for token in example.question_text.split(' '):
            all_doc_tokens.extend(tokenizer.tokenize(token))
        if len(all_doc_tokens) > max_query_length - 1:
            all_doc_tokens = all_doc_tokens[:max_query_length - 1]
        all_doc_tokens.append("[SEP]")

        for (i, token) in enumerate(example.doc_tokens):    # 遍历context的所有token（白空格分割）
            orig_to_tok_index.append(len(all_doc_tokens))   # 空格分词的token与wp分词后的token对应
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)    # wp 分词后的token对应的空格分词的token
                all_doc_tokens.append(sub_token)
            orig_to_tok_back_index.append(len(all_doc_tokens) - 1)   # 这个看意思应该是原始token与wp分词后的最后一个subtoken对应？



        def relocate_tok_span(orig_start_position, orig_end_position, orig_text):
            # word的（在para中的）开始和结束位置
            if orig_start_position is None:   # 如果输入的是none，返回0，实际上不会存在这种情况
                return 0, 0

            tok_start_position = orig_to_tok_index[orig_start_position]
            if orig_end_position < len(example.doc_tokens) - 1:   # 如果结束位置没有超出了边界
                tok_end_position = orig_to_tok_index[orig_end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1   # 超出边界
            # Make answer span more accurate.
            return _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer, orig_text)

        ans_start_position, ans_end_position = [], []
        for ans_start_pos, ans_end_pos in zip(example.start_position, example.end_position):   # 遍历每一个答案开始和结束位置
            s_pos, e_pos = relocate_tok_span(ans_start_pos, ans_end_pos, example.orig_answer_text)
            ans_start_position.append(s_pos)   # 这里返回的是答案在bert输入中的位置
            ans_end_position.append(e_pos)

        # for entity_span in example.entity_start_end_position:
        #     ent_start_position, ent_end_position \
        #         = relocate_tok_span(entity_span[0], entity_span[1], entity_span[2])
        #     entity_spans.append((ent_start_position, ent_end_position, entity_span[2], entity_span[3]))
        # 这里找到了每个实体在bert输入中的开始和结束位置

        for sent_span in example.sent_start_end_position:   # 每个句子开始和结束word的id
            if sent_span[0] >= len(orig_to_tok_index) or sent_span[0] >= sent_span[1]:
                continue  # 如果句子的开始位置大于映射表的范围，或者开始与结束位置相同（空句子），就continue
            sent_start_position = orig_to_tok_index[sent_span[0]]  # 句子在bert输入中的开始和结束位置
            sent_end_position = orig_to_tok_back_index[sent_span[1]]  # 句子结束的sub word位置（这里就是orig_to_tok_back_index的用处）
            sentence_spans.append((sent_start_position, sent_end_position))    # 句子在bert输入中的开始和结束位置

        # for para_span in example.para_start_end_position:
        #     if para_span[0] >= len(orig_to_tok_index) or para_span[0] >= para_span[1]:
        #         continue
        #     para_start_position = orig_to_tok_index[para_span[0]]
        #     para_end_position = orig_to_tok_back_index[para_span[1]]
        #     para_spans.append((para_start_position, para_end_position, para_span[2], para_span[3]))  # 3是是否是sup para

        # Padding Document
        all_doc_tokens = all_doc_tokens[:max_seq_length - 1] + ["[SEP]"]
        doc_input_ids = tokenizer.convert_tokens_to_ids(all_doc_tokens)
        query_input_ids = tokenizer.convert_tokens_to_ids(query_tokens)

        doc_input_mask = [1] * len(doc_input_ids)
        doc_segment_ids = [0] * len(query_input_ids) + [1] * (len(doc_input_ids) - len(query_input_ids))

        while len(doc_input_ids) < max_seq_length:
            doc_input_ids.append(0)
            doc_input_mask.append(0)
            doc_segment_ids.append(0)

        # Padding Question
        query_input_mask = [1] * len(query_input_ids)
        query_segment_ids = [0] * len(query_input_ids)

        while len(query_input_ids) < max_query_length:
            query_input_ids.append(0)
            query_input_mask.append(0)
            query_segment_ids.append(0)

        assert len(doc_input_ids) == max_seq_length
        assert len(doc_input_mask) == max_seq_length
        assert len(doc_segment_ids) == max_seq_length
        assert len(query_input_ids) == max_query_length
        assert len(query_input_mask) == max_query_length
        assert len(query_segment_ids) == max_query_length

        sentence_spans = get_valid_spans(sentence_spans, max_seq_length)
        # para_spans = get_valid_spans(para_spans, max_seq_length)

        sup_fact_ids = example.sup_fact_id
        sent_num = len(sentence_spans)
        sup_fact_ids = [sent_id for sent_id in sup_fact_ids if sent_id < sent_num]
        if len(sup_fact_ids) != len(example.sup_fact_id):
            failed += 1
        if example.qas_id <10:
            print("qid {}".format(example.qas_id))
            print("all_doc_tokens {}".format(all_doc_tokens))
            print("doc_input_ids {}".format(doc_input_ids))
            print("doc_input_mask {}".format(doc_input_mask))
            print("doc_segment_ids {}".format(doc_segment_ids))
            print("query_tokens {}".format(query_tokens))
            print("query_input_ids {}".format(query_input_ids))
            print("query_input_mask {}".format(query_input_mask))
            print("query_segment_ids {}".format(query_segment_ids))
            # print("para_spans {}".format(para_spans))
            print("sentence_spans {}".format(sentence_spans))
            # print("entity_spans {}".format(entity_spans))
            print("sup_fact_ids {}".format(sup_fact_ids))
            print("ans_type {}".format(ans_type))
            print("tok_to_orig_index {}".format(tok_to_orig_index))
            print("ans_start_position {}".format(ans_start_position))
            print("ans_end_position {}".format(ans_end_position))

        features.append(
            InputFeatures(qas_id=example.qas_id,
                          doc_tokens=all_doc_tokens,
                          doc_input_ids=doc_input_ids,
                          doc_input_mask=doc_input_mask,
                          doc_segment_ids=doc_segment_ids,
                          query_tokens=query_tokens,
                          query_input_ids=query_input_ids,
                          query_input_mask=query_input_mask,
                          query_segment_ids=query_segment_ids,
                          sent_spans=sentence_spans,
                          sup_fact_ids=sup_fact_ids,
                          ans_type=ans_type,
                          token_to_orig_map=tok_to_orig_index,
                          start_position=ans_start_position,
                          end_position=ans_end_position)
        )
    return features


def _largest_valid_index(spans, limit):
    for idx in range(len(spans)):
        if spans[idx][1] >= limit:
            return idx


def get_valid_spans(spans, limit):
    new_spans = []
    for span in spans:
        if span[1] < limit:
            new_spans.append(span)
        else:
            new_span = list(span)
            new_span[1] = limit - 1
            new_spans.append(tuple(new_span))
            break
    return new_spans


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return new_start, new_end

    return input_start, input_end


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--example_output", required=True, type=str)
    parser.add_argument("--feature_output", required=True, type=str)

    # Other parameters
    parser.add_argument("--do_lower_case", default=True, action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                             "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--batch_size", default=15, type=int, help="Batch size for predictions.")
    parser.add_argument("--full_data", type=str, required=True)   # 原始数据集文件
    parser.add_argument('--tokenizer_path',type=str,required=True)


    args = parser.parse_args()
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)
    examples = read_examples( full_file=args.full_data)
    with gzip.open(args.example_output, 'wb') as fout:
        pickle.dump(examples, fout)

    features = convert_examples_to_features(examples, tokenizer, max_seq_length=512, max_query_length=50)
    with gzip.open(args.feature_output, 'wb') as fout:
        pickle.dump(features, fout)











