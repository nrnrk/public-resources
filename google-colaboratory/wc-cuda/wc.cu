extern "C" __global__ void count_wc(char *text, int *line_count, int *word_count, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;

    char c = text[idx];
    char prev = (idx == 0) ? ' ' : text[idx - 1];

    // Count lines
    if (c == '\n') {
        // Must be atomic because multiple threads can increment at the same time
        atomicAdd(line_count, 1);
    }

    // Count words (transition from ' ', '\n', or '\t' to character)
    if ((prev == ' ' || prev == '\n' || prev == '\t') && (c != ' ' && c != '\n' && c != '\t')) {
        atomicAdd(word_count, 1);
    }
}
