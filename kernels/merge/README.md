## Merge Two Sorted Arrays

This kernel merges two sorted arrays into a single sorted array.

## Basic Idea

Giving two sorted arrays A and B, the goal is to produce a single sorted array C that contains all elements from A and B in sorted order. 

Sequential implementation of merging two sorted arrays involves comparing the smallest unmerged elements of both arrays and repeatedly selecting the smaller element to add to the merged array until all elements from both arrays have been processed.
It seems unable to parallelize this process directly due to its sequential nature.

We can divide the output array C into segments that can be processed in parallel. The n-th thread will be responsible for merging a specific n-th segment of the output array C.

If thread `t` is responsible for merging the segment of C from index `k`, it needs to determine the index of the first element from A and B that will contribute to this segment.

### Finding the Merge Ranges

Give an index `k` in the output array C, we need to determine the start index `i` of A and start index `j` of B that contribute to the elements of C from `C[k]`.

```c++
if (a[i] < b[j]) {
    C[k++] = a[i++];
} else {
    C[k++] = b[j++];
}
```

We can do this using a binary search approach:

```c
int findMergeIndex(int* a, int m, int* b, int n, int k) {
    int lo = max(0, k - n); // n is the size of array B
    int hi = min(k, m);     // m is the size of array A
    
    while (lo < hi) {
        int i = (lo + hi) / 2;
        int j = k - i;
        
        if (i > 0 && j < n && a[i - 1] > b[j]) {
            hi = i;
        } else if (j > 0 && i < m && b[j - 1 > a[i]) {
            lo = i + 1;
        } else {
            return i; // Found the correct partition
        }
    }
    return lo;
}
```


### Merge

If thread `t` is responsible for merging the segment of C from index `i` for `a` and index 'j' of `b`, it can merge the two segments as follows:

```c
if (i >= m) {
    c[k++] = b[j++];
} else if (j >= n) {
    c[k++] = a[i++];
} else if (a[i] < b[j]) {
    c[k++] = a[i++];
} else {
    c[k++] = b[j++];
}
```

