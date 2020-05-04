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