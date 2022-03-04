class CollinsSpan:

    def __init__(self, i, j, k, h, score):
        self.i = i
        self.j = j
        self.k = k
        self.h = h
        self.score = score

    def __str__(self):
        return "[%s, %s, %s, %s, %s]" % (self.i, self.j, self.k, self.h, self.score)

class CollinsParser:

    def __init__(self):
        self.chart = None

    def parse(self, words, pos_tags):
        self.words = words
        self.init_spans(words)
        self.heads = [None for i in range(len(words))]

        # merge spans in a bottom-up manner
        for l in range(1, len(words)+1):
            for i in range(0, len(words)):
                j = i + l
                if j > len(words): break
                for k in range(i+1, j):
                    for h_l in range(i, k):
                        for h_r in range(k, j):
                            span_l = self.chart[i][k][h_l]
                            span_r = self.chart[k][j][h_r]
                            # l -> r
                            score = self.get_score(pos_tags, span_l, span_r)
                            span = CollinsSpan(i, j, k, h_l, score)
                            self.add_span(span)
                            # r -> l
                            score = self.get_score(pos_tags, span_r, span_l)
                            span = CollinsSpan(i, j, k, h_r, score)
                            self.add_span(span)
        # top best score
        self.top_best = self.find_best(0, len(words))
        self.backtrace(self.top_best)

    # trace to left & right
    def backtrace(self, span):
        self.trace_dir(span, 'left')
        self.trace_dir(span, 'right')

    # trace direction
    def trace_dir(self, up_span, dir):
        if dir == 'left':
            current_span = self.find_best(up_span.i, up_span.k)
        elif dir == 'right':
            current_span = self.find_best(up_span.k, up_span.j)

        # decide heads
        if up_span.h != current_span.i:
            self.heads[current_span.i] = up_span.h
        elif up_span.h != current_span.j - 1:
            self.heads[current_span.j - 1] = up_span.h

        if current_span.j - current_span.i > 1:
            self.backtrace(current_span)
        else:
            pass

    def init_spans(self, words):
        # initialize chart as 3-dimensional list
        length = len(words) + 1
        chart = []
        for i in range(length):
            chart.append([])
            for j in range(length):
                chart[i].append([None] * length)
        self.chart = chart

        # add 1-length spans to the chart
        for i in range(0, len(words)):
            span = CollinsSpan(i, i+1, i, i, 0.0)
            self.add_span(span)

    def add_span(self, new_span):
        i, j, h = new_span.i, new_span.j, new_span.h
        old_span = self.chart[i][j][h]
        if old_span is None or old_span.score < new_span.score:
            self.chart[i][j][h] = new_span # update chart

    def get_score(self, pos_tags, head, dep):

        # scoring function with POS tags
        head_pos = pos_tags[head.h]
        dep_pos = pos_tags[dep.h]

        if head_pos == "V" and dep_pos == "N":
            score = 3.0
        elif head_pos == "N" and dep_pos == "DT":
            score = 1.0
        elif head_pos == "N" and dep_pos == "JJ":
            score = 1.0
        elif head_pos == "V" and dep_pos == "PR":
            score = 2.0
        else:
            score = 0.1

        # calculate score based on arc-factored model
        return head.score + dep.score + score

    """ Find the highest-scored span [i, j, h] from [i, j] """
    def find_best(self, i, j):
        best_span = None
        for h in range(i, j):
            span = self.chart[i][j][h]
            if best_span is None or best_span.score < span.score:
                best_span = span
        return best_span

    def print_heads(self):
        head_words = []
        for i in self.heads:
            try:
                head_words.append(self.words[i])
            except TypeError:
                #head_words.append(None)
                head_words.append('ROOT')
        print(head_words)

if __name__ == '__main__':
    # run
    p = CollinsParser()
    words = ["She", "read", "a", "short", "novel"]
    pos_tags = ["PR", "V", "DT", "JJ", "N"]
    p.parse(words, pos_tags)
    p.print_heads()