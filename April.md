# May LeetCoding Challenge

1) https://leetcode.com/problems/single-number/ </br>
Given a non-empty array of integers, every element appears twice except for one. Find that single one.
Note:
Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory?
```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        
        res = 0
        for num in nums:
            res = res ^ num
            
        return res
            
        
```
### Complexity: O(n) , space: O(1)
----------------------
2) https://leetcode.com/problems/happy-number/ </br>
Write an algorithm to determine if a number n is "happy".
A happy number is a number defined by the following process: Starting with any positive integer, replace the number by the sum of the squares of its digits, and repeat the process until the number equals 1 (where it will stay), or it loops endlessly in a cycle which does not include 1. Those numbers for which this process ends in 1 are happy numbers.
Return True if n is a happy number, and False if not.

```python
class Solution(object):
    def isHappy(self, n):
        """
        :type n: int
        :rtype: bool
        """
        
        x = 1
        seen = {}
        
        while n != 1:
            
            summ = 0

            while n != 0:
                summ += (n % 10) ** 2
                n = n // 10

            if summ in seen:
                break
            else:
                seen[summ] = 0
                n = summ

        if n == 1:
            return True
        else:
            return False
```
### Complexity: O(don't sure) , space: O(n)
-----------------------

3) https://leetcode.com/problems/maximum-subarray/ </br>
Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        
        if not nums:
            return 0
        
        length = len(nums)
        
        dp = [0] * length
        dp[0] = nums[0]
        
        for i in range(1, length):
            dp[i] = max(nums[i], dp[i-1] + nums[i])
            
        return max(dp)
       
```
### Complexity: O(n) , space: O(n)
-----------------------

4) https://leetcode.com/problems/move-zeroes/ </br>
Given an array nums, write a function to move all 0's to the end of it while maintaining the relative order of the non-zero elements.

```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        length = len(nums)
        zero_pos = 0
        i = 0

        while i < length:
            
            if nums[zero_pos] == 0:
                if nums[i] != 0:
                    nums[zero_pos], nums[i] = nums[i], nums[zero_pos]
                    zero_pos += 1
            else:
                zero_pos += 1
                
            i += 1  
```
### Complexity: O(n) , space: O(1)
-----------------------
5) https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/ </br>
Say you have an array prices for which the ith element is the price of a given stock on day i.
Design an algorithm to find the maximum profit. You may complete as many transactions as you like (i.e., buy one and sell one share of the stock multiple times).
Note: You may not engage in multiple transactions at the same time (i.e., you must sell the stock before you buy again)

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        
        length = len(prices)
        
        if length == 1:
            return 0
        
        profit = 0
        
        for i in range(length - 1):
            if prices[i+1] > prices[i]:
                profit += prices[i+1] - prices[i]

        return profit
```
### Complexity: O(n) , space: O(1)
-----------------------
6) https://leetcode.com/problems/group-anagrams/ </br>
Given an array of strings, group anagrams together.<br>
Example:<br>
Input: ["eat", "tea", "tan", "ate", "nat", "bat"]<br>
Output:
[
  ["ate","eat","tea"],
  ["nat","tan"],
  ["bat"]
]<br>
Note:<br>
All inputs will be in lowercase.<br>
The order of your output does not matter.

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        
        dic = {}
        
        for word in strs:
            s = ''.join(sorted(word))
            if s in dic:
                dic[s].append(word)
            else:
                dic[s] = [word]
                
        result = []
        
        for k,v in dic.items():
            result.append(v)
            
        return result
        
```
### Complexity: O( nlen(word)log(len(word)) ) , space: O(n)
-----------------------
7) https://leetcode.com/problems/counting-elements/ </br>
Given an integer array arr, count how many elements x there are, such that x + 1 is also in arr.
If there're duplicates in arr, count them seperately. order of your output does not matter.

```python
class Solution:
    def countElements(self, arr: List[int]) -> int:
        
        dict_elements = {elm:0 for elm in arr}
        
        counter = 0
        
        for elm in arr:
            if elm+1 in dict_elements:
                counter += 1
                
        return counter
             
```
### Complexity: O(n) , space: O(n)
-----------------------
8) https://leetcode.com/problems/middle-of-the-linked-list/ </br>
Given a non-empty, singly linked list with head node head, return a middle node of linked list.
If there are two middle nodes, return the second middle node.
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def middleNode(self, head: ListNode) -> ListNode:
               
        # Can be solved by 2 pointers instead and it will be in one pass 
        
        if not head:
            return head
        
        ptr = head
        nodes_counter = 1
        
        while ptr.next:
            nodes_counter += 1
            ptr = ptr.next
            
        pos = nodes_counter // 2
        
        ptr = head
        while pos != 0:
            pos -= 1
            ptr = ptr.next
            
        return ptr
```
### Complexity: O(n) , space: O(1)
-----------------------
9) https://leetcode.com/problems/backspace-string-compare/ </br>
Given two strings S and T, return if they are equal when both are typed into empty text editors. # means a backspace character.

Note that after backspacing an empty text, the text will continue empty.
```python
class Solution(object):
    def backspaceCompare(self, S, T):
        """
        :type S: str
        :type T: str
        :rtype: bool
        """
        new_s = ''
        new_t = ''
        
        for char in S:
            if char == '#':
                if  len(new_s) != 0:
                    new_s = new_s[0:-1]
            else:
                new_s = new_s + char
                
        for char in T:
            if char == '#':
                if len(new_t) != 0:
                    new_t = new_t[0:-1]
            else:
                new_t = new_t + char
                
        return new_t == new_s
```
### Complexity: O(n) , space: O(n)
-----------------------
10) https://leetcode.com/problems/min-stack/ </br>
Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.</br></br>
push(x) -- Push element x onto stack.</br>
pop() -- Removes the element on top of the stack.</br>
top() -- Get the top element.</br>
getMin() -- Retrieve the minimum element in the stack.
 
```python
class Node:
     def __init__(self, val):
        """
        initialize your data structure here.
        """
        self.val = val
        self.next = None
        self.minimum_till_now = None
        self.previous = None
        
class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.head = None
        self.ptr = self.head
    
    def push(self, x: int) -> None:
        
        node = Node(x)
        
        if not self.head:
            node.minimum_till_now = x
            self.head = node
            self.ptr = self.head
        else:
            node.minimum_till_now = min(x, self.ptr.minimum_till_now)
            self.ptr.next = node
            prev = self.ptr
            self.ptr = self.ptr.next
            self.ptr.previous = prev

    def pop(self) -> None:
        if self.head and self.head.next: 
            self.ptr = self.ptr.previous
            self.ptr.next = None

        else:
            self.head = None

    def top(self) -> int:
        return self.ptr.val

    def getMin(self) -> int:
        return self.ptr.minimum_till_now


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()
```
### Complexity: O(1) , space: O(n)
-----------------------
11) https://leetcode.com/problems/diameter-of-binary-tree/ </br>
Given a binary tree, you need to compute the length of the diameter of the tree. The diameter of a binary tree is the length of the longest path between any two nodes in a tree. This path may or may not pass through the root.
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        
        diameter = 0

        def traverse_tree(root):
            nonlocal diameter
            if root:
                l = traverse_tree(root.left)
                r = traverse_tree(root.right)
                summation = l + r
                if summation > diameter:
                    diameter = summation
                return max(l,r) + 1
            else:
                return 0
            
        traverse_tree(root)
        
        return diameter
        
```
### Complexity: O(#nodes) , space: O(#nodes)
-----------------------
12) https://leetcode.com/problems/last-stone-weight/ </br>
We have a collection of stones, each stone has a positive integer weight.
Each turn, we choose the two heaviest stones and smash them together.  Suppose the stones have weights x and y with x <= y.  The result of this smash is:
- If x == y, both stones are totally destroyed;
- If x != y, the stone of weight x is totally destroyed, and the stone of weight y has new weight y-x.
- At the end, there is at most 1 stone left.  Return the weight of this stone (or 0 if there are no stones left.)
```python
class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:
        
        stones.sort()
        
        while len(stones) > 1:
            
            if stones[-2] == stones[-1]:
                # remove 2 elements
                stones = stones[:-2]
            else:
                # remove 1 elements
                stones[-2] = stones[-1] - stones[-2]
                stones[-1] -= stones[-1]
                stones = stones[:-1]
                stones.sort() # we can use insertion sort instead because python uses quick sort
            
        return stones[0] if stones else 0      
            
```
### Complexity: O(n**2) , space: O(1)
-----------------------
13) https://leetcode.com/problems/contiguous-array/  </br>
Given a binary array, find the maximum length of a contiguous subarray with equal number of 0 and 1.
```python
class Solution:
    def findMaxLength(self, nums: List[int]) -> int:
        
        count_to_index = {}
        
        count = 0
        maximum = 0
        
        for ind,num in enumerate(nums):
            if num == 1:
                count += 1
            else:
                count -= 1
            
            if count == 0:
                maximum = max(maximum, ind+1)
                continue
                
            if count not in count_to_index:
                count_to_index[count] = ind
            else:
                maximum = max(maximum, ind - count_to_index[count])
  
        return maximum
         
            
```
### Complexity: O(n) , space: O(n)
-----------------------
14) https://leetcode.com/problems/perform-string-shifts/  </br>
You are given a string s containing lowercase English letters, and a matrix shift, where shift[i] = [direction, amount]:<br>
- direction can be 0 (for left shift) or 1 (for right shift). 
- amount is the amount by which string s is to be shifted.
- A left shift by 1 means remove the first character of s and append it to the end.
- Similarly, a right shift by 1 means remove the last character of s and add it to the beginning.
Return the final string after all operations.
```python
class Solution:
    def stringShift(self, s: str, shift: List[List[int]]) -> str:
        
        shifts = 0
        length = len(s)
        
        for shi in shift:
            if shi[0] == 1:
                shifts += shi[1]
            else:
                shifts -= shi[1]
                
        if shifts == 0:
            return s
        
        # shift right
        elif shifts > 0:
            shifts = shifts % length
            s = s[-shifts:] + s[:-shifts]
            
        # shift left
        else:
            shifts *= - 1
            shifts = shifts % length
            s = s[shifts:] + s[:shifts]
            
        return s
            
```
### Complexity: O(n) , space: O(1)
-----------------------
15) https://leetcode.com/problems/product-of-array-except-self/  </br>
Given an array nums of n integers where n > 1,  return an array output such that output[i] is equal to the product of all the elements of nums except nums[i].
```python
class Solution:
    class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        
        length = len(nums)
        
        if length == 2:
            return nums[::-1]
                
        left_product = []

        prod = 1
        for num in nums:
            left_product.append(prod)
            prod *= num
                
        prod = 1
        for i in range(length-1, -1, -1):
            left_product[i] *= prod
            prod *= nums[i]
            
            
        return left_product
            
```
### Complexity: O(n) , space: O(1)
-----------------------
17) https://leetcode.com/problems/number-of-islands/  </br>
Given a 2d grid map of '1's (land) and '0's (water), count the number of islands. An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.
```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        
        if not grid:
            return 0
        
        # Padding the matrix
        rows = len(grid)
        cols = len(grid[0])
        
        for i in range(rows):
            grid[i].insert(0, '0')
            grid[i].append('0')
              
        rows += 2
        cols += 2
                
        grid.insert(0, ['0'] * (cols))
        grid.append(['0'] * (cols))
        
        ###########################################
        
        one_pos = [(i,j) for i in range(rows) for j in range(cols) if grid[i][j] == '1']
        visited = {}
        
        counter = 0
        
        for pos in one_pos:
            if pos in visited:
                continue
            else:
                # BFS
                queue = [pos]
                counter += 1
                while queue:
                    r,c = queue.pop(0)
                    for i,j in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:
                        if grid[i][j] == '0':
                            continue
                        elif (i,j) not in visited:
                            visited[(i,j)] = 0
                            queue.append((i,j))
                    grid[r][c] = '0'
         
        return counter
                
```
### Complexity: O(n*m) , space: O(n*m)
-----------------------
18) https://leetcode.com/problems/minimum-path-sum/  </br>
Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right which minimizes the sum of all numbers along its path.

Note: You can only move either down or right at any point in time.
```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        
        if not grid:
            return 0
                
        num_rows = len(grid)
        num_cols = len(grid[0])
        
        visited = {}
        queue = [[(0,0), grid[0][0]]]
        
        while queue:
            tup,sum_till_now = queue.pop(0)
            r,c = tup
            
            for i,j in [(r,c+1), (r+1,c)]:
                if i < num_rows and j < num_cols:
                    if (i,j) in visited:
                        queue[-1][-1] = min(queue[-1][-1], sum_till_now + grid[i][j])
                    else:
                        queue.append([(i,j), sum_till_now + grid[i][j]])
                        visited[(i,j)] =  sum_till_now + grid[i][j]
            # print(queue)
        return sum_till_now
          
```
### Complexity: O(n+m) , space: O(n+m)
-----------------------
19) https://leetcode.com/problems/search-in-rotated-sorted-array/  </br>
- Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.
(i.e., [0,1,2,4,5,6,7] might become [4,5,6,7,0,1,2]).
- You are given a target value to search. If found in the array return its index, otherwise return -1.
- You may assume no duplicate exists in the array.
- Your algorithm's runtime complexity must be in the order of O(log n).

Example 1:
```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        
        l = 0
        r = len(nums)-1
        
        while l<=r:
            mid = (l+r) // 2
            
            if nums[mid] == target:
                return mid
            if nums[l] == target:
                return l
            if nums[r] == target:
                return r
            
            if nums[mid] > target:
                # from l to m is sorted or from m to r is sorted
                if target > nums[l] or nums[r] > nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1
            
            else:
                # from l to m is sorted or from m to r is sorted
                if target < nums[r] or nums[l] < nums[mid]:
                    l = mid + 1
                else:
                    r = mid - 1
            
        return -1
          
```
### Complexity: O(log n) , space: O(1)
-----------------------
20) https://leetcode.com/problems/construct-binary-search-tree-from-preorder-traversal/  </br>
- Return the root node of a binary search tree that matches the given preorder traversal.
- (Recall that a binary search tree is a binary tree where for every node, any descendant of node.left has a value < node.val, and any descendant of node.right has a value > node.val.  Also recall that a preorder traversal displays the value of the node first, then traverses node.left, then traverses node.right.)
- It's guaranteed that for the given test cases there is always possible to find a binary search tree with the given requirements.

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def bstFromPreorder(self, preorder: List[int]) -> TreeNode:
        
        if not preorder:
            return None
        
        root_val = preorder[0]
        root = TreeNode(root_val)
        ptr = root
        stack = [root]
        
        for num in preorder[1:]:
            
            node = TreeNode(num)
            
            if num < root_val:
                ptr.left = node
                stack.append(node)
                ptr = ptr.left
            else:
                while stack and num > stack[-1].val:
                    right = stack.pop()
                right.right = node
                ptr = node
                stack.append(node)

            root_val = num

        return root  
```
### Complexity: O(n) , space: O(n)
-----------------------
21) https://leetcode.com/problems/leftmost-column-with-at-least-a-one/  </br>
- binary matrix means that all elements are 0 or 1. For each individual row of the matrix, this row is sorted in non-decreasing order.
- Given a row-sorted binary matrix binaryMatrix, return leftmost column index(0-indexed) with at least a 1 in it. If such index doesn't exist, return -1.
- You can't access the Binary Matrix directly.  You may only access the matrix using a BinaryMatrix interface:
- BinaryMatrix.get(row, col) returns the element of the matrix at index (row, col) (0-indexed).
- BinaryMatrix.dimensions() returns a list of 2 elements [rows, cols], which means the matrix is rows * cols.
- Submissions making more than 1000 calls to BinaryMatrix.get will be judged Wrong Answer.  Also, any solutions that attempt to circumvent the judge will result in disqualification.

```python
# """
# This is BinaryMatrix's API interface.
# You should not implement it, or speculate about its implementation
# """
#class BinaryMatrix(object):
#    def get(self, x: int, y: int) -> int:
#    def dimensions(self) -> list[]:

class Solution:
    def leftMostColumnWithOne(self, binaryMatrix: 'BinaryMatrix') -> int:
        
        r,c = binaryMatrix.dimensions()
        
        start_row = 0
        start_col = c - 1
        
        while start_col >= 0 and start_row < r :
            last = binaryMatrix.get(start_row,start_col)
            if last:
                start_col -= 1
            else:
                start_row += 1
                
        return -1 if not last and start_row == r and start_col == c-1 else start_col+1 
```
### Complexity: O(n+m) , space: O(1)
-----------------------
22) https://leetcode.com/problems/subarray-sum-equals-k/  </br>
Given an array of integers and an integer k, you need to find the total number of continuous subarrays whose sum equals to k
```python
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        
        counter = 0
        su = 0
        sum_to_frequency = {0:1}
        
        for num in nums:
            su += num
            diff =  su - k
            if diff in sum_to_frequency:
                counter += sum_to_frequency[diff]
                
            if su in sum_to_frequency:
                sum_to_frequency[su] += 1
            else:
                sum_to_frequency[su] = 1
        return counter
    
```
### Complexity: O(n) , space: O(n)
-----------------------
23) https://leetcode.com/problems/bitwise-and-of-numbers-range/  </br>
Given a range [m, n] where 0 <= m <= n <= 2147483647, return the bitwise AND of all numbers in this range, inclusive.
```python
class Solution:
    def rangeBitwiseAnd(self, m: int, n: int) -> int:
        
        n_in_binary = bin(n)[2:]
        n_length = len(n_in_binary)
        
        m_in_binary = bin(m)[2:]
        m_length = len(m_in_binary)
        
        if n_length > m_length:
            return 0
        else:
            res = m
            for i in range(m, n+1):
                res &= i
            return res
    
```
### Complexity: O(n-m) , space: O(n)
-----------------------
24) https://leetcode.com/problems/lru-cache/  </br>
- Design and implement a data structure for Least Recently Used (LRU) cache. It should support the following operations: get and put.
- get(key) - Get the value (will always be positive) of the key if the key exists in the cache, otherwise return -1.
- put(key, value) - Set or insert the value if the key is not already present. When the cache reached its capacity, it - should invalidate the least recently used item before inserting a new item.
- The cache is initialized with a positive capacity.
```python
from collections import OrderedDict 

class LRUCache:

    def __init__(self, capacity: int):
        self.lru = OrderedDict()
        self.cap = capacity

    def get(self, key: int) -> int:
        if key in self.lru:
            value = self.lru.pop(key)
            self.lru[key] = value
            return value
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.lru:
            _ = self.get(key)
        elif self.cap == 0:
            self.lru.popitem(last=False)
        else:
            self.cap -= 1
        self.lru[key] = value

# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```
### Complexity: O(1) , space: O(n)
-----------------------
25) https://leetcode.com/problems/jump-game/  </br>
- Given an array of non-negative integers, you are initially positioned at the first index of the array.
- Each element in the array represents your maximum jump length at that position.
- Determine if you are able to reach the last index
```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        
        length = len(nums)
        max_reach = 0
        
        for ind,num in enumerate(nums):
            if max_reach < ind:
                return False
            elif max_reach >= length-1:
                return True
            max_reach = max(max_reach, ind+num)
```
### Complexity: O(n) , space: O(1)
-----------------------
28) https://leetcode.com/problems/first-unique-number/  </br>
- You have a queue of integers, you need to retrieve the first unique integer in the queue.
- Implement the FirstUnique class:
- FirstUnique(int[] nums) Initializes the object with the numbers in the queue.
- int showFirstUnique() returns the value of the first unique integer of the queue, and returns -1 if there is no such integer.
- void add(int value) insert value to the queue.
```python
class node:
     def __init__(self, val):
        """
        initialize your data structure here.
        """
        self.val = val
        self.next = None
        self.prev = None
        
class FirstUnique:

    def __init__(self, nums: List[int]):
        self.head = None
        self.ptr = self.head
        self.num_frequency = {}
        
        self.num_counter = collections.Counter(nums)
        for num in nums:
            if self.num_counter[num] == 1:
                self.num_frequency[num] = node(num)
                self.__insert(self.num_frequency[num])
            
    def showFirstUnique(self) -> int:
        if self.head:
            return self.head.val
        return -1

    def add(self, value: int) -> None:
        if value in self.num_frequency:
            self.__delete(self.num_frequency[value])
        elif value in self.num_counter:
            return 
        else:
            self.num_frequency[value] = node(value)
            self.__insert(self.num_frequency[value])
        
    def __insert(self, node):
        if not self.head:
            self.head = node
            self.ptr = self.head
            
        else:
            self.ptr.next = node
            node.prev = self.ptr
            self.ptr = self.ptr.next
            
    def __delete(self, node):
        
        if node.prev:
            ptr = node.prev
            ptr.next = node.next
            if node.next:
                node.next.prev = ptr
        else:
            if node.next:
                node.next.prev = None
            self.head = node.next

# Your FirstUnique object will be instantiated and called as such:
# obj = FirstUnique(nums)
# param_1 = obj.showFirstUnique()
# obj.add(value)
```
### Complexity: O(1 for each method) , space: O(n)
-----------------------