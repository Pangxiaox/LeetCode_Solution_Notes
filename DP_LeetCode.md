# 动态规划

### 最长上升子序列

```java
public class LIS {
    public int lengthOfLIS(int [] nums)
    {
        int length = nums.length;
        int [] dp = new int[length];
        if(length == 0)
            return 0;
        for(int i=0;i<dp.length;i++)
        {
            dp[i]=1;
        }

        for(int j=1;j<nums.length;j++)
        {
            for(int k=0;k<j;k++)
            {
                if(nums[k]<nums[j])
                {
                    dp[j]=Math.max(dp[k]+1,dp[j]);
                }
            }
        }

        int res=1;
        for(int a=0;a<dp.length;a++)
        {
            res = Math.max(res,dp[a]);
        }
        return res;
    }
}
```



### 最大子序和

```java
public class LSS {
    public int subMax(int[]nums)
    {
        int length=nums.length;
        int dp[]=new int[length];
        dp[0]=nums[0];
        int max=dp[0];

        for(int i=1;i<nums.length;i++)
        {
            dp[i]=nums[i]+(dp[i-1]>0?dp[i-1]:0);
            max=Math.max(dp[i],max);
        }
        return max;
    }
}
```



### 爬楼梯（斐波那契数列DP解法）

```java
public class climbStairs {
    public int climb(int n)
    {
        int dp[]=new int[n];
        if(n==1)
        {
            return 1;
        }
        dp[0]=1;
        dp[1]=2;
        for(int i=2;i<n;i++)
        {
            dp[i]=dp[i-1]+dp[i-2];
        }
        return dp[n-1];
    }
}
```



### 打家劫舍

两间相邻的房屋在同一晚上被小偷闯入，系统自动报警。给定一个代表每个房屋存放金额的非负整数数组，计算在不触动报警装置情况下能够偷窃到的最高金额。

输入：[1,2,3,1]

输出：4（1+3=4）

```java
class Solution {
    public int rob(int[] nums) {
        int len = nums.length;
        if(len == 0)
            return 0;
        int[] dp = new int[len + 1];
        dp[0] = 0;
        dp[1] = nums[0];
        for(int i = 2; i <= len; i++) {
            dp[i] = Math.max(dp[i-1], dp[i-2] + nums[i-1]);
        }
        return dp[len];
    }
}
```



### 买卖股票的最佳时机

一个数组，第i个元素是一支给定股票第i天的价格

最多允许完成一笔交易（买入和卖出一支股票），计算所能获取最大利润

输入：[7,1,5,3,6,4]

输出：5（6-1=5）

**设当前为第i天，minPrice表示前i-1天最低价格，maxProfit表示前i-1天最大收益，那么第i天最大收益=max（在第i天卖出的所得收益，前i-1天的最大收益）**

```java
    public int maxProfit(int[] prices) {
        int minPrice = Integer.MAX_VALUE;
        int maxProfit = 0;
        for (int i = 0; i < prices.length; i++) {
            minPrice = Math.min(minPrice, prices[i]);
            maxProfit = Math.max(maxProfit, prices[i] - minPrice);
        }
        return maxProfit;
    }
```



### 不同路径

一个机器人位于一个m*n网格左上角，每次只能向下或者向右走一步，达到网格右下角共有多少条路径

**dp[i] [j]表示到达[i] [j]最多路径，dp[i] [j]=dp[i-1] [j]+dp[i] [j-1],dp[0] [j]=dp[i] [0]=1**

```java
class Solution {
    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];
        for (int i = 0; i < n; i++) dp[0][i] = 1;
        for (int i = 0; i < m; i++) dp[i][0] = 1;
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[m - 1][n - 1];  
    }
}
```

优化后：

```java
class Solution {
    public int uniquePaths(int m, int n) {
        int[] cur = new int[n];
        Arrays.fill(cur,1);
        for (int i = 1; i < m;i++){
            for (int j = 1; j < n; j++){
                cur[j] += cur[j-1] ;
            }
        }
        return cur[n-1];
    }
}
```





