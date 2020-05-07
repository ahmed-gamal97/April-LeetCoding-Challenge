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
