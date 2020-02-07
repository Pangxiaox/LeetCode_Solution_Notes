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

👆最长上升子序列

👇最大子序和（53）

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

👇爬楼梯（70）🍉斐波那契数列的DP解法

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

