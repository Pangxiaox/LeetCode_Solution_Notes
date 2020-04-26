# LeetCode热题100部分题解

### 1. 找到所有数组中消失的数字

输入：[4,3,2,7,8,2,3,1]

输出：[5,6]

```java
class Solution {
    public List<Integer> findDisappearedNumbers(int[] nums) {
        List<Integer> res = new ArrayList<>();
        for(int i=0;i<nums.length;i++)
        {
           int index = Math.abs(nums[i])-1;
           if(nums[index]>0)
           {
               nums[index] *= -1;
           }
        }

        for(int i=1;i<=nums.length;i++)
        {
            if(nums[i-1]>0)
            {
                res.add(i);
            }
        }
        return res;
    }
}
```



### 2. 合并二叉树

```java
class Solution {
    public TreeNode mergeTrees(TreeNode t1, TreeNode t2) {
        if(t1 == null)  
            return t2;
        if(t2==null)
            return t1;
        t1.val +=t2.val;
        t1.left = mergeTrees(t1.left,t2.left);
        t1.right = mergeTrees(t1.right,t2.right);
        return t1;
    }
}
```

递归遍历



### 3. 移动零

```java
class Solution {
    public void moveZeroes(int[] nums) {
        if(nums==null)
            return;
        int j=0;
        for(int i=0;i<nums.length;i++)
        {
            if(nums[i]!=0)
            {
                int tmp = nums[i];
                nums[i] = nums[j];
                nums[j] = tmp;
                j++;
            }
        }
    }
}
```



### 4. 二叉搜索树转换为累加树

```java
class Solution {
    int num = 0;
    public TreeNode convertBST(TreeNode root) {
        if(root!=null)
        {
            convertBST(root.right);
            root.val += num;
            num = root.val;
            convertBST(root.left);
            return root;
        }
        return null;
    }
}
```

反向中序遍历



### 5. 打家劫舍

```java
class Solution {
    public int rob(int[] nums) {
        if(nums.length==0)
            return 0;        
        int[] dp =new int[nums.length+1];
        dp[0]=0;
        dp[1]=nums[0];
        for(int i=2;i<=nums.length;i++)
        {
            dp[i] = Math.max(dp[i-1],dp[i-2]+nums[i-1]);
        }
        return dp[nums.length];
    }
}
```



### 6. 环形链表

```java
public class Solution {
    public boolean hasCycle(ListNode head) {
        HashSet<ListNode> set = new HashSet<>();
        while(head!=null)
        {
            if(set.contains(head))
            {
                return true;
            }
            else
            {
                set.add(head);
            }
            head = head.next;
        }
        return false;
    }
}
```

哈希表遍历

```java
public class Solution {
    public boolean hasCycle(ListNode head) {
        if(head==null||head.next==null) return false;
        ListNode slow = head;
        ListNode fast = head.next;
        while(slow!=fast)
        {
            if(fast==null||fast.next==null)
            {
                return false;
            }
            slow = slow.next;
            fast = fast.next.next;
        }
        return true;
    }
}
```

快慢指针



### 7. 全排列

回溯

```java
class Solution {
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer>list = new ArrayList<>();
        backtrack(res,list,nums);
        return res;
    }

    public void backtrack(List<List<Integer>> res,List<Integer> list,int[]nums)
    {
        if(list.size()==nums.length)
        {
            res.add(new ArrayList<Integer> (list));
        }
        for(int num:nums)
        {
            if(!list.contains(num))
            {
                list.add(num);
                backtrack(res,list,nums);
                list.remove(list.size()-1);
            }
        }
    }
}

```



### 8. 子集

回溯

```java
class Solution {
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        backtrack(0, nums, res, new ArrayList<Integer>());
        return res;
    }

    private void backtrack(int i, int[] nums, List<List<Integer>> res, ArrayList<Integer> tmp) {
        res.add(new ArrayList<>(tmp));
        for (int j = i; j < nums.length; j++) {
            tmp.add(nums[j]);
            backtrack(j + 1, nums, res, tmp);
            tmp.remove(tmp.size() - 1);
        }
    }
}
```



### 9. 最长上升子序列

```java
class Solution {
    public int lengthOfLIS(int[] nums) {
        if(nums.length == 0) return 0;
        int[] dp = new int[nums.length];
        int res = 0;
        Arrays.fill(dp, 1);
        for(int i = 0; i < nums.length; i++) {
            for(int j = 0; j < i; j++) {
                if(nums[j] < nums[i]) dp[i] = Math.max(dp[i], dp[j] + 1);
            }
            res = Math.max(res, dp[i]);
        }
        return res;
    }
}
```



### 10. 路径总和

```java
class Solution {
    public boolean hasPathSum(TreeNode root, int sum) {
        if(root==null)
            return false;

        sum -=root.val;
        if(root.left==null&&root.right==null)
        {
            return (sum==0);
        }
        return hasPathSum(root.left,sum)||hasPathSum(root.right,sum);
    }
}
```



### 11. 编辑距离

二维dp

```java
class Solution {
    public int minDistance(String word1, String word2) {
        int n1 = word1.length();
        int n2 = word2.length();
        int[][]dp = new int[n1+1][n2+1];

        for(int i=1;i<=n1;i++)
        {
            dp[i][0]=dp[i-1][0]+1;
        }
        for(int j=1;j<=n2;j++)
        {
            dp[0][j]=dp[0][j-1]+1;
        }

        for(int i=1;i<=n1;i++)
        {
            for(int j=1;j<=n2;j++)
            {
                if(word1.charAt(i-1)==word2.charAt(j-1))
                {
                    dp[i][j]=dp[i-1][j-1];
                }
                else
                {
                    dp[i][j]=Math.min(dp[i-1][j],Math.min(dp[i][j-1],dp[i-1][j-1]))+1;
                }
            }
        }
        return dp[n1][n2];
    }
}
```



### 12. 最长回文子串

```java
class Solution{
    public String longestPalindrome(String s) {
        if (s == null || s.length() < 2) {
            return s;
        }
        int strLen = s.length();
        int maxStart = 0;  //最长回文串的起点
        int maxEnd = 0;    //最长回文串的终点
        int maxLen = 1;  //最长回文串的长度

        boolean[][] dp = new boolean[strLen][strLen];

        for (int r = 1; r < strLen; r++) {
            for (int l = 0; l < r; l++) {
                if (s.charAt(l) == s.charAt(r) && (r - l <= 2 || dp[l + 1][r - 1])) {
                    dp[l][r] = true;
                    if (r - l + 1 > maxLen) {
                        maxLen = r - l + 1;
                        maxStart = l;
                        maxEnd = r;
                    }
                }
            }
        }
        return s.substring(maxStart, maxEnd + 1);
    }
}
```



### 13. 用队列实现一个栈

```java
class MyStack {

    Queue<Integer> queue;
    
    public MyStack() {
        queue = new LinkedList<>();
    }
    
    public void push(int x) {
        queue.add(x);
        for(int i = 1; i < queue.size(); i++)
            queue.add(queue.remove());
    }
    
    public int pop() {
        return queue.poll();
    }
    
    public int top() {
        return queue.peek();
    }

    public boolean empty() {
        return queue.size() == 0;
    }
}
/**
 * Your MyStack object will be instantiated and called as such:
 * MyStack obj = new MyStack();
 * obj.push(x);
 * int param_2 = obj.pop();
 * int param_3 = obj.top();
 * boolean param_4 = obj.empty();
 */
```



### 14. 零钱兑换

```java
public class Solution {
  public int coinChange(int[] coins, int amount) {
    int max = amount + 1;
    int[] dp = new int[amount + 1];
    Arrays.fill(dp, max);
    dp[0] = 0;
    for (int i = 1; i <= amount; i++) {
      for (int j = 0; j < coins.length; j++) {
        if (coins[j] <= i) {
          dp[i] = Math.min(dp[i], dp[i - coins[j]] + 1);
        }
      }
    }
    return dp[amount] > amount ? -1 : dp[amount];
  }
}
```

