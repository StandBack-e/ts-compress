import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    t = int(input[ptr])
    ptr += 1
    for _ in range(t):
        n = int(input[ptr])
        ptr += 1
        a = input[ptr]
        ptr += 1
        # 预处理前缀和数组，prefix0[i]表示前i个字符（a[0]到a[i-1]）中'0'的数量
        # prefix1[i]表示前i个字符中'1'的数量
        prefix0 = [0] * (n + 1)
        prefix1 = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix0[i] = prefix0[i-1] + (1 if a[i-1] == '0' else 0)
            prefix1[i] = prefix1[i-1] + (1 if a[i-1] == '1' else 0)
        best_i = 0
        min_diff = float('inf')
        for i in range(n + 1):
            # 计算左侧需要的最少'0'数量：ceil(i / 2)
            left_need = (i + 1) // 2
            if prefix0[i] < left_need:
                continue
            # 计算右侧需要的最少'1'数量：ceil((n - i) / 2)
            right_count = prefix1[n] - prefix1[i]
            right_need = ((n - i) + 1) // 2
            if right_count < right_need:
                continue
            # 计算当前i与中间位置的差的绝对值
            diff = abs(i - n / 2)
            # 更新最优解：差更小，或差相等时i更小
            if diff < min_diff or (diff == min_diff and i < best_i):
                min_diff = diff
                best_i = i
        print(best_i)

if __name__ == "__main__":
    main()