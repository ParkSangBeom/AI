import numpy as np

class DataConverter:

    data = []
    input_data = []
    output_data = []

    def __init__(self, path):

        input = []
        output = []

        with open(path, 'r', encoding='utf-8') as content_file:
            index = 0
            line_data = []

            for line in content_file:
                line = line.replace("\n", "")
                self.data.extend(line)
                line_data.append(line)

                if index % 2 == 0:
                    input.append(line) # 질문.
                else:           
                    output.append(line) # 답.
                    print(line)

                index += 1

                #if index == 2:
                #    break;

            self.data_max_length = len(max(line_data, key=len))
            self.index_to_char = list(set(self.data)) # index -> char
            self.char_to_index = {c : i for i, c in enumerate(self.index_to_char)} # char -> idex
            self.index_size = len(self.index_to_char)

            self.input_data = self.StrToOnehot(input)
            self.output_data = self.StrToOnehot(output)

    def StrToOnehot(self, data):
        r_data = []
        for i in data:
            zero_pad = np.full_like(np.empty(self.data_max_length), self.char_to_index[" "])
            #zero_pad = np.zeros(self.data_max_length)
            i_str = [self.char_to_index[c] for c in i]

            for k in range(len(i_str)):
                zero_pad[k] = i_str[k]

            r_data.append(zero_pad)

        return r_data

    #def OneHot(self, data):
    #    r_data = []
    #    for i in data:
    #        onehot = np.zeros(self.index_size, dtype = np.int)
    #        onehot[i] = 1
    #        r_data.append(onehot)

    #    return np.array(r_data)