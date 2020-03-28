# 剑指Offer题解

### 数组中重复的数字

```java
class Solution {
    public int findRepeatNumber(int[] nums) {
        Arrays.sort(nums);
        int ans = 0;
        for(int i=0;i<nums.length-1;i++)
        {
            if(nums[i]==nums[i+1])
            {
                ans = nums[i];
            }
        }
        return ans;
    }
}
```

**推荐解：**

```java
class Solution {
    public int findRepeatNumber(int[] nums) {
        Set<Integer> numsSet = new HashSet<>();
        for(int num: nums) {
            if(!numsSet.add(num)) {
                return num;
            }
        }
        return -1;
    }
}
```



### 二维数组中的查找

从左下角开始

```java
class Solution {
    public boolean findNumberIn2DArray(int[][] matrix, int target) {
        int rows = matrix.length;
        if(rows == 0)
            return false;

        int columns = matrix[0].length;
        if(columns == 0)
            return false;

        int x = rows - 1;
        int y = 0;

        while(x>=0)
        {
            while(y<columns && matrix[x][y]<target)
            {
                y++;
            }     
            if(y<columns && matrix[x][y] == target)
            {
                return true;
            }
            x--;
        }   
        return false;
    }
}
```



### 替换空格

```java
class Solution {
    public String replaceSpace(String s) {
        return s.replaceAll(" ","%20");
    }
}
```



### 从尾到头打印链表

先遍历一次链表获取链表长度，再遍历一次打印链表，从数组尾部开始放值

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public int[] reversePrint(ListNode head) {
        ListNode headnode = head;
        int length = 0;
        while(headnode!=null)
        {
            length++;
            headnode = headnode.next;           
        }
        int arr[] = new int[length];
        while(length>=1)
        {
            arr[length-1] = head.val;
            head = head.next;
            length--;
        }
        return arr;
    }
}
```



### 求1+2+...+n

等差数列求和，右移运算符表示除以2

```java
class Solution {
    public int sumNums(int n) {
        return ((int)(Math.pow(n,2)+n))>>1;
    }
}
```



### 第一个只出现一次的字符

```java
class Solution {
   public char firstUniqChar(String s) {
        int[] hash = new int[123];
        for (int i = 0; i < s.length(); i++) {
            hash[(int) s.charAt(i)]++;
        }
        for (int i = 0; i < s.length(); i++) {
            if (hash[(int)s.charAt(i)] == 1) {
                return (char) s.charAt(i);
            }
        }
        return ' ';
    }
}
```



### 数组中出现次数超过一半的数字

数组排序，中间的数字就是答案

```java
class Solution {
    public int majorityElement(int[] nums) {
        Arrays.sort(nums);
        return nums[nums.length/2];
    }
}
```



### 最小的k个数

```java
class Solution {
    public int[] getLeastNumbers(int[] arr, int k) {
        Arrays.sort(arr);
        return Arrays.copyOf(arr,k);
    }
}
```



### 打印从1到最大的n位数

```java
class Solution {
    public int[] printNumbers(int n) {
        int[]res=new int[(int)Math.pow(10,n)-1];
        for(int i=0;i<Math.pow(10,n)-1;i++)
        {
            res[i]=i+1;
        }
        return res;
    }
}
```



### 和为s的两个数字

双指针，利用递增排序数组特点

```java
class Solution {
    public int[] twoSum(int[] nums, int target) {
        int res[]=new int[2];
        int l=0;
        int r=nums.length-1;
        while(l<r)
        {
            if(nums[l]+nums[r]<target)
            {
                l++;
            }
            else if(nums[l]+nums[r]>target)
            {
                r--;
            }
            else
            {
                res[0]=nums[l];
                res[1]=nums[r];
                break;
            }
        }
        return res;
    }
}
```



### 0——n-1中缺失的数字

```java
class Solution {
    public int missingNumber(int[] nums) {
        for(int i=0;i<nums.length;i++)
        {
            if(nums[i]!=i)
                return i;
        }
        return nums.length;
    }
}
```



### 圆圈中最后剩下的数字

推导公式，约瑟夫环问题

```java
class Solution {
    public int lastRemaining(int n, int m) {
        int cnt=0;
        for(int i=2;i<n+1;i++)
        {
            cnt = (cnt+m)%i;
        }
        return cnt;
    }
}
```



### 在排序数组中查找数字

```java
class Solution {
    public int search(int[] nums, int target) {
        int res=0;
        for(int i=0;i<nums.length;i++)
        {
            if(nums[i]==target)
                res++;
        }
        return res;
    }
}
```





### 数组中数字出现的次数Ⅰ

HashSet，找出两个只出现一次的数字

```java
class Solution {
    public int[] singleNumbers(int[] nums) {
        HashSet<Integer> set = new HashSet<>();
        for(int num:nums)
        {
            if(set.contains(num))
                set.remove(num);
            else
                set.add(num);
        }
        return set.stream().mapToInt(Integer::intValue).toArray();
    }
}
```



### 数组中数字出现的次数Ⅱ

除一个数字只出现一次，其他数字都出现了三次，找出只出现了一次的数字

```java
class Solution {
    public int singleNumber(int[] nums) {
        HashMap<Integer,Integer> map = new HashMap<>();
        for(int num:nums)
        {
            map.put(num,map.getOrDefault(num,0)+1);
        }
        for(int num:nums)
        {
            if(map.get(num)==1)
                return num;
        }
        return 0;
    }
}
```



### 调整数组顺序使奇数位于偶数前面

```java
class Solution {
    public int[] exchange(int[] nums) {
        int l=0;
        int r=nums.length-1;
        while(l<r)
        {
            while(l<r&&nums[l]%2==1)
            {
                l++;
            }
            while(l<r&&nums[r]%2==0)
            {
                r--;
            }
            int tmp=nums[l];
            nums[l]=nums[r];
            nums[r]=tmp;
            l++;
            r--;
        }
        return nums;
    }
}
```



### 二叉树的深度

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public int maxDepth(TreeNode root) {
        if(root==null)
            return 0;
        return 1+Math.max(maxDepth(root.left),maxDepth(root.right));
    }
}
```



### 二叉树的镜像

递归

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public TreeNode mirrorTree(TreeNode root) {
        if(root==null)
            return null;
        TreeNode tmp=root.left;
        root.left=root.right;
        root.right=tmp;
        mirrorTree(root.left);
        mirrorTree(root.right);
        return root;
    }
}
```



### 链表中倒数第k个节点

双指针

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode getKthFromEnd(ListNode head, int k) {
        ListNode slow = head;
        ListNode fast = head;
        for(int i=0;i<k;i++)
        {
            fast=fast.next;
        }
        while(fast!=null)
        {
            slow=slow.next;
            fast=fast.next;
        }
        return slow;
    }
}
```



### 反转链表

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode reverseList(ListNode head) {
        ListNode tmp =null;
        ListNode cur=head;
        ListNode pre=null;
        while(cur!=null)
        {
            tmp=cur.next;
            cur.next=pre;
            pre=cur;
            cur=tmp;
        }
        return pre;
    }
}
```



### 合并两个排序链表

```java
class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode head = new ListNode(-1);
        ListNode tmpHead = head;
        while(l1 != null && l2 != null){
            if(l1.val < l2.val){
                tmpHead.next = l1;
                l1 = l1.next;
            }else{
                tmpHead.next = l2;
                l2 = l2.next;
            }
            tmpHead = tmpHead.next;
        }
        tmpHead.next = l1 == null? l2 : l1;
        return head.next;
    }
}
```



### 二进制中1的个数

位运算

```java
public class Solution {
    // you need to treat n as an unsigned value
    public int hammingWeight(int n) {
        int cnt=0;
        while(n!=0)
        {
            cnt++;
            n = n&(n-1);
        }
        return cnt;
    }
}
```



### 左旋转字符串

```java
class Solution {
    public String reverseLeftWords(String s, int n) {
     return s.substring(n,s.length())+s.substring(0,n);
    }
}
```



### 从上到下打印二叉树Ⅱ

层序遍历

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> result = new LinkedList<>();
        if(root==null)
            return result;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while(!queue.isEmpty())
        {
            int size = queue.size();
            List<Integer> tmp = new LinkedList<>();
            for(int i=0;i<size;i++)
            {
                TreeNode node = queue.poll();
                tmp.add(node.val);
                if(node.left!=null)
                {
                    queue.add(node.left);
                }
                if(node.right!=null)
                {
                    queue.add(node.right);
                }
            }
            result.add(tmp);
        }
        return result;
    }
}
```



### 两个链表的第一个公共节点

双指针

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if(headA==null||headB==null)
            return null;
        ListNode pA=headA;
        ListNode pB=headB;
        while(pA!=pB)
        {
            pA = pA == null ? headB : pA.next;
            pB = pB == null ? headA : pB.next;
        }
        return pA;
    }
}
```



### 删除链表的节点

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode deleteNode(ListNode head, int val) {
        ListNode tmp = new ListNode(0);
        tmp.next=head;
        ListNode node = tmp;
        while(node.next!=null)
        {
            if(node.next.val == val)
            {
                node.next = node.next.next;
                break;
            }
            node = node.next;
        }
        return tmp.next;
    }
}
```



### 构建乘积数组

左乘右乘

```java
class Solution {
    public int[] constructArr(int[] a) {
        int[]b = new int[a.length];
        int v =1;
        for(int i=0;i<a.length;i++)
        {
            b[i]=v;
            v*=a[i];
        }
        v=1;
        for(int i=a.length-1;i>=0;i--)
        {
            b[i]*=v;
            v*=a[i];
        }
        return b;
    }
}
```



### 旋转数组的最小数字

```java
class Solution {
    public int minArray(int[] numbers) {
     Arrays.sort(numbers);
     return numbers[0];
    }
}
```

**推荐：**

二分法

```java
public class Solution {

    // [3, 4, 5, 1, 2]
    // [1, 2, 3, 4, 5]
    // 不能使用左边数与中间数比较，这种做法不能有效地减治

    // [1, 2, 3, 4, 5]
    // [3, 4, 5, 1, 2]
    // [2, 3, 4, 5 ,1]

    public int minArray(int[] numbers) {
        int len = numbers.length;
        if (len == 0) {
            return 0;
        }
        int left = 0;
        int right = len - 1;
        while (left < right) {
            int mid = (left + right) >>> 1;
            if (numbers[mid] > numbers[right]) {
                // [3, 4, 5, 1, 2]，mid 以及 mid 的左边一定不是最小数字
                // 下一轮搜索区间是 [mid + 1, right]
                left = mid + 1;
            } else if (numbers[mid] == numbers[right]) {
                // 只能把 right 排除掉，下一轮搜索区间是 [left, right - 1]
                right = right - 1;
            } else {
                // 此时 numbers[mid] < numbers[right]
                // mid 的右边一定不是最小数字，mid 有可能是，下一轮搜索区间是 [left, mid]
                right = mid;
            }
        }

        // 最小数字一定在数组中，因此不用后处理
        return numbers[left];
    }
}
```



### 青蛙跳台阶

斐波那契数列、DP

```java
class Solution {
    public int numWays(int n) {
       int[]dp=new int[101];
       dp[0]=1;
       dp[1]=1;
       dp[2]=2;
       for(int i=3;i<=100;i++)
       {
           dp[i]=(dp[i-1]+dp[i-2])%1000000007;
       }
       return dp[n];
    }
}
```



### 斐波那契数列

F（0）=0，F（1）=1

```java
class Solution {
    public int fib(int n) {
        int[]dp=new int[101];
        dp[0]=0;
        dp[1]=1;
        for(int i=2;i<=100;i++)
        {
            dp[i]=(dp[i-1]+dp[i-2])%1000000007;
        }
        return dp[n];
    }
}
```



### 两个栈实现队列

```java
class CQueue {
     private Stack<Integer> stack1 ;
     private Stack<Integer> stack2 ;
    
     public CQueue() {
        stack1 = new Stack<Integer>();
        stack2 = new Stack<Integer>();
    }
    
     public void appendTail(int value) {
        stack1.push(value);
        return;
    }
    
     public int deleteHead() {
        if(stack2.empty()){
            if(stack1.empty()){
            return -1; 
            }else{
                while(!stack1.empty()){
                stack2.push(stack1.pop());
                }
            }
        }
            return stack2.pop();
    }
}

/**
 * Your CQueue object will be instantiated and called as such:
 * CQueue obj = new CQueue();
 * obj.appendTail(value);
 * int param_2 = obj.deleteHead();
 */
```



### 对称的二叉树

二叉树如果和它的镜像一样，那么就是对称的

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public boolean isSymmetric(TreeNode root) {
        return isSymmetricHelper(root,root);
    }

    public boolean isSymmetricHelper(TreeNode root1,TreeNode root2)
    {
        if(root1==null&&root2==null)
            return true;
        if(root1==null||root2==null)
            return false;
        if(root1.val!=root2.val)
            return false;
        return isSymmetricHelper(root1.left,root2.right) && isSymmetricHelper(root1.right,root2.left);
    }
}
```



### 平衡二叉树

```java
class Solution {
    public boolean isBalanced(TreeNode root) {
        //递归三部曲，二叉树的题目大部分都可以使用递归
        //1.找终止条件，树为空的时候即无需继续递归
        return depth(root) != -1;
        //2.找返回值，返回的应该是自己是否是BST以及左右子树的差值
        //3.一次递归应该做什么，左右子树的BST都是true，且要判断最后一次是否是BST
    }

    public static int depth(TreeNode root){
        if(root == null) return 0;
        int left = depth(root.left);
        if(left == -1) return -1;
        int right = depth(root.right);
        if(right == -1) return -1;
        return Math.abs(left - right) < 2 ? Math.max(left,right) + 1 : -1;
    }
}
```



### 不用加减乘除做加法

a+b相当于（a^b)+((a&b)<<1)

```java
class Solution {
    public int add(int a, int b) {
        while(a!=0)
        {
            int tmp = a ^ b;
            a = ((a&b)<<1);
            b = tmp;
        }
        return b;
    }
}
```



### 数值的整数次方

```java
class Solution {
    public double myPow(double x, int n) {
        if(n==0) return 1;
        if(n==1) return x;
        if(n==-1) return 1/x;
        double half = myPow(x,n/2);
        double mod = myPow(x,n%2);
        return half*half*mod;
    }
}
```



### 丑数

DP

丑数 = 2^x * 3^y * 5^z ,x、y、z可以为0

```java
class Solution {
    public int nthUglyNumber(int n) {
        int[] res = new int[n];
        res[0]=1;
        int p2=0,p3=0,p5=0;
        for(int i=1;i<n;i++)
        {
            res[i]=Math.min(res[p2]*2,Math.min(res[p3]*3,res[p5]*5));
            if(res[i]==res[p2]*2) p2++;
            if(res[i]==res[p3]*3) p3++;
            if(res[i]==res[p5]*5) p5++;
        }
        return res[n-1];
    }
}
```



### 股票的最大利润

DP

```java
class Solution {
    public int maxProfit(int[] prices) {
        int minPrice = Integer.MAX_VALUE;
        int maxProfit = 0;
        for(int i=0;i<prices.length;i++)
        {
            minPrice = Math.min(minPrice,prices[i]);
            maxProfit = Math.max(maxProfit,prices[i]-minPrice);
        }
        return maxProfit;
    }
}
```



### 把数组排成最小的数

把数组所有数字拼接起来排成一个数

```java
class Solution {
    public String minNumber(int[] nums) {
        ArrayList<String> list =new ArrayList<>();
        StringBuffer sb = new StringBuffer();
        for(int i:nums)
        {
            list.add(String.valueOf(i));
        }
        list.sort((o1,o2) -> (o1+o2).compareTo(o2+o1));
        for(String s:list)
        {
            sb.append(s);
        }
        return sb.toString();
    }
}
```



### 礼物的最大价值

DP，原地二维数组修改

```java
class Solution {
    public int maxValue(int[][] grid) {
        int row=grid.length;
        int col=grid[0].length;
        for(int i=1;i<row;i++)
        {
            grid[i][0]+=grid[i-1][0];
        }
        for(int j=1;j<col;j++)
        {
            grid[0][j]+=grid[0][j-1];
        }
        for(int i=1;i<row;i++)
        {
            for(int j=1;j<col;j++)
            {
                grid[i][j]+=Math.max(grid[i][j-1],grid[i-1][j]);
            }
        }
        return grid[row-1][col-1];
    }
}
```



### 包含min函数的栈

实现一个能够得到栈的最小元素的min函数

```java
class MinStack {
    Stack<Integer> stack;
    Stack<Integer> min;
    /** initialize your data structure here. */
    public MinStack() {
        stack=new Stack<>();
        min=new Stack<>();
    }
    
    public void push(int x) {
        stack.push(x);
        if(min.isEmpty())
        {
            min.push(x);
        }
        else
        {
            int tmp=x>min.peek() ? min.peek() : x;
            min.push(tmp);
        }
    }
    
    public void pop() {
        stack.pop();
        min.pop();
    }
    
    public int top() {
        return stack.peek();
    }
    
    public int min() {
        return min.peek();
    }
}

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack obj = new MinStack();
 * obj.push(x);
 * obj.pop();
 * int param_3 = obj.top();
 * int param_4 = obj.min();
 */
```



### 顺时针打印矩阵

```java
class Solution {
    public int[] spiralOrder(int[][] matrix) {
        int row = matrix.length;
        if(row==0)
        {
            return new int[]{};
        }
        int col = matrix[0].length;
        int cnt=0;
        int left=0,right=col-1,top=0,bottom=row-1;
        int[]res = new int[row*col];
        while(top<=bottom&&right>=left)
        {
            for(int i=left;i<=right;i++)
            {
                res[cnt++]=matrix[top][i];
            }
            top++;
            for(int i=top;i<=bottom;i++)
            {
                res[cnt++]=matrix[i][right];
            }
            right--;
            for(int i=right;i>=left&&top<=bottom;i--)
            {
                res[cnt++]=matrix[bottom][i];
            }
            bottom--;
            for(int i=bottom;i>=top&&left<=right;i--)
            {
                res[cnt++]=matrix[i][left];
            }
            left++;
        }
        return res;
    }
}
```



### 栈的压入、弹出序列

```java
class Solution {
    public boolean validateStackSequences(int[] pushed, int[] popped) {
        Stack<Integer> stack = new Stack<>();
        int pushIndex = 0;
        for (int poppedIndex = 0; poppedIndex < popped.length; ++poppedIndex) {
            while (pushIndex < pushed.length && (stack.empty() || stack.peek() != popped[poppedIndex]))
                stack.push(pushed[pushIndex++]);
            if (stack.peek() != popped[poppedIndex])
                return false;
            else
                stack.pop();
        }
        return true;
    }
}
```



### 连续子数组最大和

DP

```java
class Solution {
    public int maxSubArray(int[] nums) {
        int res = nums[0];
        int max = nums[0];
        for(int i=1;i<nums.length;i++)
        {
            if(max<0)
                max = nums[i];
            else
                max += nums[i];
            res = Math.max(res,max);
        }
        return res;
    }
}
```



### n个骰子的点数

DP

```java
class Solution {
    public double[] twoSum(int n) {
        int [][]dp = new int[n+1][6*n+1];
        //边界条件
        for(int s=1;s<=6;s++)dp[1][s]=1;
        for(int i=2;i<=n;i++){
            for(int s=i;s<=6*i;s++){
                //求dp[i][s]
                for(int d=1;d<=6;d++){
                    if(s-d<i-1)break;//为0了
                    dp[i][s]+=dp[i-1][s-d];
                }
            }
        }
        double total = Math.pow((double)6,(double)n);
        double[] ans = new double[5*n+1];
        for(int i=n;i<=6*n;i++){
            ans[i-n]=((double)dp[n][i])/total;
        }
        return ans;
    }
}
```



### 剪绳子

DP

n>3时，若除以3后余数为1则将3和1换为2和2，解中最多包含两个2，因此先计算3的数量，再计算2的数量

```java
class Solution {
    public int cuttingRope(int n) {
        int num_2 = 0, num_3 = 0;
        if(n==2)
            return 1;
        if(n==3)
            return 2;
        num_3 = n/3;
        int remain = n % 3;
        if(remain==1)
        {
            num_3--;
            remain+=3;
        }
        num_2 = remain /2;
        double result = Math.pow(3,num_3)*Math.pow(2,num_2);
        return (int) result;
    }
}
```



### 剪绳子Ⅱ

数据范围变大，int和long都用不了，不适合DP，采取取模运算的特质

```java
class Solution {
    public int cuttingRope(int n) {
       if(n == 2)
            return 1;
        if(n==3)
            return 2;
        long res = 1;
        while(n>4)
        {
            res*=3;
            res%=1000000007;
            n-=3;
        } 
        return (int)(res*n%1000000007);
    }
}
```



### 二叉树中和为某一值的路径

链表+回溯

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution { 
    private List<List<Integer>> res = new LinkedList<>();
    private LinkedList<Integer> list = new LinkedList<>();
    private int target;

    public List<List<Integer>> pathSum(TreeNode root, int sum) {
       target = sum;
       dfs(root,0);
       return res;
    }

    public void dfs(TreeNode root,int cur_sum)
    {
        if(root!=null)
        {
            int val = root.val;
            list.addLast(val);
            cur_sum+=val;
            if(cur_sum==target&&root.left==null&&root.right==null)
                res.add(new LinkedList<>(list));
            else
            {
                dfs(root.left,cur_sum);
                dfs(root.right,cur_sum);
            }
            list.removeLast();
        }
    }
}
```



### 把数字翻译成字符串

DP，上楼梯问题升级版

```java
class Solution {
    public int translateNum(int num) {
        String str=num+"";
        int length=str.length();
        if(length==0)
            return 0;
        if(length==1)
            return 1;
        int[]dp=new int[length+1];
        dp[0]=1;
        dp[1]=1;
        for(int i=2;i<length+1;i++)
        {
            if(str.charAt(i-2)!='0'&&((str.charAt(i-2)-'0')*10+str.charAt(i-1)-'0')<26)
                dp[i]=dp[i-1]+dp[i-2];
            else
                dp[i]=dp[i-1];
        }
        return dp[length];
    }
}
```



### 表示数值的字符串

```java
class Solution {
    public boolean isNumber(String s) {
        if(s == null || s.trim().length() == 0) return false;
        s = s.trim();
        int[] idx = {0};
        boolean flag = judgeNumber(s, idx);
        if(idx[0] != s.length() && s.charAt(idx[0]) == '.'){
            idx[0]++;
            //System.out.println(idx[0]);
            flag = judgeUnsignedNumber(s, idx) || flag;
        }
        if(idx[0] != s.length() && (s.charAt(idx[0]) == 'e' || s.charAt(idx[0]) == 'E')){
            idx[0]++;
            flag = flag && judgeNumber(s, idx);
        }
        return flag && idx[0] == s.length();
    }
    private boolean judgeNumber(String s, int[] idx){
        if(idx[0] >= s.length()){
            return false;
        }
        if(s.charAt(idx[0]) == '+' || s.charAt(idx[0]) == '-'){
            idx[0]++;
        }
        return judgeUnsignedNumber(s, idx);
    }

    private boolean judgeUnsignedNumber(String s, int[] idx){
        
        int before = idx[0];
        while(idx[0] < s.length() && s.charAt(idx[0]) <= '9' && s.charAt(idx[0]) >= '0'){
            idx[0]++;
        }
        //System.out.println(before + ":" + idx[0]);
        return idx[0] == before? false : true;
    }
}
```



### 数字序列中某一位的数字

```java
class Solution {
    public int findNthDigit(int n) {
        if (n < 10) {
            return n;
        }
        int base = 1;
        long count = 0;  //计算有多少位,测试的时候发现有个1e9的用例，这个用例会导致count越界
        while (true) {
            count = helper(base);
            if (n < count) break;
            n -= count;
            base++;
        }
        //得到新的n和count了，算出第n位对应什么数字
        int num = (int) (n / base + Math.pow(10, base - 1));
        return String.valueOf(num).charAt(n % base) - '0';
    }

    // 计算当前有多少位 1位数10种，10位；2位数 90个数字 180位；3位数 900个 2700位
    private long helper(int base) {
        if (base == 1) {
            return 10;
        }
        return (long) (Math.pow(10, base - 1) * 9 * base);
    }
}
```



### 二叉搜索树的后序遍历序列

```java
class Solution{
    public boolean verifyPostorder(int [] postorder) {
        if (postorder.length <= 2) return true;
        return verifySeq(postorder, 0, postorder.length-1);
    }
    private boolean verifySeq(int[] postorder, int start, int end) {
        if (start >= end) return true;
        int i;
        for (i = start; i < end; i++) {
            if (postorder[i] > postorder[end]) break;
        }
        // 验证后面的是否都大于sequence[end]
        for (int j = i; j < end; j++) {
            if (postorder[j] < postorder[end]) return false;
        }
        return verifySeq(postorder, start, i-1) && verifySeq(postorder, i, end-1);
    }
}
```



### 矩阵中的路径

```java
class Solution {
    boolean res = false;
    public boolean exist(char[][] board, String word) {
        int length1 = board.length;
        int length2 = board[0].length;
        boolean[][] isSearched = new boolean[length1][length2];
        
        for(int i=0; i<length1; i++)
        {
            for(int k=0; k<length2; k++)
            {
                DFS(isSearched,board,i,k,word,0,length1,length2);
            }
                
        }
        
        return res;
    }
    
    public void DFS(boolean[][] isSearched,char[][] board,
                    int loc1,int loc2,String word,int target,
                   int length1,int length2)
    {
        if(loc1<0 || loc1>=length1 || loc2<0 || loc2>=length2 )
            return ;
        if(isSearched[loc1][loc2] || res || target>=word.length())
            return ;
        
        if(board[loc1][loc2] == word.charAt(target))
        {
            if(target==word.length()-1)
            {
                res = true;
                return ;
            }
            else
            {
                isSearched[loc1][loc2] = true;
                DFS(isSearched,board,loc1-1,loc2,word,target+1,length1,length2);
                DFS(isSearched,board,loc1+1,loc2,word,target+1,length1,length2);
                DFS(isSearched,board,loc1,loc2-1,word,target+1,length1,length2);
                DFS(isSearched,board,loc1,loc2+1,word,target+1,length1,length2);
                isSearched[loc1][loc2] = false;
            }
        }
        else
        {
            return ;
        }
    }
}
```



### 最长不含重复字符的子字符串

滑动窗口

```java
class Solution{
    public int lengthOfLongestSubstring(String s) {
        Set<Character> set = new HashSet<>();
        int left = 0, right = 0, max = 0;
        while(right < s.length()) {
            while(set.contains(s.charAt(right))) {
                set.remove(s.charAt(left));
                left++;
            }
            set.add(s.charAt(right));
            right++;
            max = Math.max(right - left, max);
        }
        return max;
    }
}
```



### 正则表达式匹配

```java
class Solution {
    public boolean isMatch(String s, String p) {
        int sLength = s.length(), pLength = p.length();
        boolean[][] dp = new boolean[sLength + 1][pLength + 1];
        dp[sLength][pLength] = true;
        for (int i = sLength; i >= 0; i--) {
            for (int j = pLength - 1; j >= 0; j--) {
                boolean charMatch = i < sLength && (s.charAt(i) == p.charAt(j) || p.charAt(j) == '.');
                if (j < pLength - 1 && p.charAt(j + 1) == '*')
                    dp[i][j] = dp[i][j + 2] || charMatch && dp[i + 1][j];
                else
                    dp[i][j] = charMatch && dp[i + 1][j + 1];
            }
        }
        return dp[0][0];
    }
}
```



### 数组中的逆序对

```java
public class Solution {

    // 后有序数组中元素出列的时候，计算逆序个数

    public int reversePairs(int[] nums) {
        int len = nums.length;
        if (len < 2) {
            return 0;
        }
        int[] temp = new int[len];
        return reversePairs(nums, 0, len - 1, temp);
    }

    /**
     * 计算在数组 nums 的索引区间 [left, right] 内统计逆序对
     *
     * @param nums  待统计的数组
     * @param left  待统计数组的左边界，可以取到
     * @param right 待统计数组的右边界，可以取到
     * @return
     */
    private int reversePairs(int[] nums, int left, int right, int[] temp) {
        // 极端情况下，就是只有 1 个元素的时候，这里只要写 == 就可以了，不必写大于
        if (left == right) {
            return 0;
        }

        int mid = (left + right) >>> 1;

        int leftPairs = reversePairs(nums, left, mid, temp);
        int rightPairs = reversePairs(nums, mid + 1, right, temp);

        int reversePairs = leftPairs + rightPairs;
        if (nums[mid] <= nums[mid + 1]) {
            return reversePairs;
        }

        int reverseCrossPairs = mergeAndCount(nums, left, mid, right, temp);
        return reversePairs + reverseCrossPairs;

    }

    /**
     * [left, mid] 有序，[mid + 1, right] 有序
     * @param nums
     * @param left
     * @param mid
     * @param right
     * @param temp
     * @return
     */
    private int mergeAndCount(int[] nums, int left, int mid, int right, int[] temp) {
        // 复制到辅助数组里，帮助我们完成统计
        for (int i = left; i <= right; i++) {
            temp[i] = nums[i];
        }

        int i = left;
        int j = mid + 1;
        int res = 0;
        for (int k = left; k <= right; k++) {
            if (i > mid) {
                // i 用完了，只能用 j
                nums[k] = temp[j];
                j++;
            } else if (j > right) {
                // j 用完了，只能用 i
                nums[k] = temp[i];
                i++;
            } else if (temp[i] <= temp[j]) {
                // 此时前数组元素出列，不统计逆序对
                nums[k] = temp[i];
                i++;
            } else {
                // 此时后数组元素出列，统计逆序对，快就快在这里，一次可以统计出一个区间的个数的逆序对
                nums[k] = temp[j];
                j++;
                res += (mid - i + 1);
            }
        }
        return res;
    }
}
```



### 序列化二叉树

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
public class Codec {

    //很明显是一个层序遍历，
    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        if(root == null)return "[]";
        String res = "[";
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while(!queue.isEmpty()){
            TreeNode cur = queue.poll();
            if(cur!=null){
                res+=cur.val+",";
                queue.offer(cur.left);
                queue.offer(cur.right);
            }else{
                res+="null,";
            }
        }
        //去除最后的一个,
        res = res.substring(0,res.length()-1);
        return res+="]";
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        if(data == null || "[]".equals(data))return null;
        String res = data.substring(1,data.length()-1);
        String[] values = res.split(",");
        int index = 0;
        TreeNode head = generateTreeNode(values[index++]);
        Queue<TreeNode> queue = new LinkedList<>();
        TreeNode node = head;
        queue.offer(head);
        while(!queue.isEmpty()){
            node = queue.poll();
            node.left = generateTreeNode(values[index++]);
            node.right = generateTreeNode(values[index++]);
            if(node.left!=null){
                queue.offer(node.left);
            }
            if(node.right!=null){
                queue.offer(node.right);
            }
        }
        return head;
    }
    public TreeNode generateTreeNode(String value){
        if("null".equals(value))return null;
        return new TreeNode(Integer.valueOf(value));
    }
}

// Your Codec object will be instantiated and called as such:
// Codec codec = new Codec();
// codec.deserialize(codec.serialize(root));
```



### 数据流中的中位数

```java
class MedianFinder {

    private PriorityQueue<Integer> maxHeap, minHeap;

    /** initialize your data structure here. */
    public MedianFinder() {
        maxHeap = new PriorityQueue<>(Collections.reverseOrder());
        minHeap = new PriorityQueue<>();
    }

    public void addNum(int num) {
        maxHeap.offer(num);
        minHeap.offer(maxHeap.poll());
        //如果不平衡则调整
        if (minHeap.size() > maxHeap.size()) {
            maxHeap.offer(minHeap.poll());
        }
    }

    public double findMedian() {
        if (maxHeap.size() == minHeap.size()) {
            return (maxHeap.peek() + minHeap.peek()) * 0.5;
        }
        return maxHeap.peek();
    }
}

/**
 * Your MedianFinder object will be instantiated and called as such:
 * MedianFinder obj = new MedianFinder();
 * obj.addNum(num);
 * double param_2 = obj.findMedian();
 */
```



### 二叉搜索树的第k大节点

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public int kthLargest(TreeNode root, int k) {
        List<Integer> result = new LinkedList<>();
        Stack<TreeNode> stack = new Stack<>();
        TreeNode cur = root;
        while (cur != null || !stack.isEmpty()) {
            while (cur != null) {
                stack.push(cur);
                cur = cur.right;
            }
            cur = stack.pop();
            result.add(cur.val);
            cur = cur.left;
        }
        return result.get(k - 1);
    }
}
```



### 二叉搜索树的最近公共祖先

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(p.val==root.val || q.val==root.val){
            return root;
        }
        if(p.val>q.val){
            if(p.val>root.val && q.val<root.val){
               return root;
            }

        }else if(p.val<q.val){
            if(p.val<root.val && q.val>root.val){
                return root;
            }

        }else{
            return p;
        }
        if(p.val<root.val){
            return lowestCommonAncestor(root.left,p,q);
        }else{
            return lowestCommonAncestor(root.right,p,q);
        }
    }
}
```



### 二叉树的最近公共祖先

```java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root == null || p == root || q == root)return root;
        TreeNode left = lowestCommonAncestor(root.left,p,q);
        TreeNode right = lowestCommonAncestor(root.right,p,q);
        if(left!=null && right != null)return root;
        return left == null ? right : left;
    }
}
```



### 和为s的连续正数序列

```java
class Solution{
public int[][] findContinuousSequence(int target) {
    int i = 1; // 滑动窗口的左边界
    int j = 1; // 滑动窗口的右边界
    int sum = 0; // 滑动窗口中数字的和
    List<int[]> res = new ArrayList<>();

    while (i <= target / 2) {
        if (sum < target) {
            // 右边界向右移动
            sum += j;
            j++;
        } else if (sum > target) {
            // 左边界向右移动
            sum -= i;
            i++;
        } else {
            // 记录结果
            int[] arr = new int[j-i];
            for (int k = i; k < j; k++) {
                arr[k-i] = k;
            }
            res.add(arr);
            // 左边界向右移动
            sum -= i;
            i++;
        }
    }
        return res.toArray(new int[res.size()][]);
    }
}
```



### 翻转单词顺序

```java
class Solution {
    public String reverseWords(String s) {
        String[] str = s.trim().split("\\s+");
        StringBuilder sb = new StringBuilder();
        for(int i=str.length-1;i>0;i--)
        {
            sb.append(str[i]);
            sb.append(" ");
        }
        sb.append(str[0]);
        return sb.toString();
    }
}
```



### 滑动窗口的最大值

```java
class Solution{
public int[] maxSlidingWindow(int[] nums, int k) {
        int n,index=0;
        n=nums.length+1-k;
        int[] re=new int[n];
        Queue<Integer> req=new LinkedList<Integer>();
        int max=Integer.MIN_VALUE;
        if(nums==null || nums.length==0) {
        	return new int[0];
        }
        if(k==1) {
        	return nums;
        }
        for(int j=0;j<k;j++) {
    		req.offer(nums[j]);
    		if(nums[j]>max) {
    			max=nums[j];
    		}
    	}
        re[index]=max;
        index+=1;
        for(int i=0;i<nums.length-k;i++) {
        	int flag=req.peek();
        	int f=i+k;
        	req.poll();
        	if(flag<max) {
        		if(nums[f]>max) {
        			max=nums[f];
        		}
        		re[index]=max;
    			index+=1;
    			req.offer(nums[f]);
        	}else {
    			req.offer(nums[f]);
    			max=Integer.MIN_VALUE;
        		for(int q:req) {
        			if(q>max) {
        				max=q;
        			}
        		}
        		re[index]=max;
    			index+=1;
        	}
        }
        return re;
    }
}
```



### 扑克牌中的顺子

```java
class Solution {
    public boolean isStraight(int[] nums) {
        int max = 0;
        int min = 20;
        int zeroNum = 0;
        Set<Integer> nonZeroNum = new HashSet<Integer>();
        for(int x: nums){
            if(x != 0){
                max = max > x ? max : x;
                min = min < x ? min : x;
                nonZeroNum.add(x);
            }else{
                zeroNum += 1;
            }
        }
        // 最大值和最小值之间的差距小于等于4，同时0的数目和其他非重复的数目应该大于0
        if(max - min <= 4 && zeroNum + nonZeroNum.size() == 5){
            return true;
        }

        return false;
    }
}
```



### 复杂链表的复制

```java
/*
// Definition for a Node.
class Node {
    int val;
    Node next;
    Node random;

    public Node(int val) {
        this.val = val;
        this.next = null;
        this.random = null;
    }
}
*/
class Solution {
    public Node copyRandomList(Node head) {
        if(head==null){
            return null;
        }
        copy(head);
        randomDirect(head);
        return reList(head);
    }
    //拷贝链表
    private void copy(Node head){
        while(head!=null){
            Node cloneNode = new Node(head.val);
            Node nextNode = head.next;
            head.next = cloneNode;
            cloneNode.next = nextNode;
            head = cloneNode.next;
        }
    }
    //指定随机指针
    private void randomDirect(Node head){
        while(head!=null){
            Node cloneNode = head.next;
            if(head.random!=null){
                Node direct = head.random;
                cloneNode.random = direct.next;
            }
            head = cloneNode.next;
        }
    }
    //重新连接 链表
    private Node reList(Node head){
        Node cloneNode = head.next;
        Node cloneHead = cloneNode;
        head.next = cloneNode.next;
        head = head.next;
        while(head!=null){
            cloneNode.next = head.next;
            head.next = head.next.next;
            head = head.next;
            cloneNode = cloneNode.next;
        }
        return cloneHead;
    }
}
```



### 重建二叉树

给定前序遍历数组和中序遍历数组

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        /*
		 * 创建只有一个元素的数组，该元素的值指向前序序列的元素，之所以用数组，是为了
		 * 能让通过引用让递归函数都能访问到，保证preorder数组不重复遍历
		 */
		int[] current = {0}; 
		return build(current, 0, inorder.length-1,preorder, inorder);
    }

    /**
	 * @param current：存放着一个记录当前遍历到前序序列的位置的元素
	 * @param start：切割后得到的序列的起始位置下标
	 * @param end：切割后得到的序列的结束位置下标
	 * @param preorder：前序序列
	 * @param inorder：中序序列
	 * @return：返回当前节点
	 */
    private TreeNode build(int[] current, int start, int end, int[] preorder, int[] inorder) {
		if(start > end) return null; //如果起始下标大于结束下标则返回null，表示没有子节点了
		TreeNode node = new TreeNode(preorder[current[0]]); //根据当前前序序列中的值创建一个节点
		int j= start;
		for(;j<=end;j++) {
			//如果切割后的序列中找到与当前节点等值的节点才跳出循环
			if(inorder[j] == preorder[current[0]]) break; 
		}
		current[0]++; //前序序列中当前所在位置后移一位，进行下面的递归函数
		node.left = build(current, start, j-1, preorder, inorder);
		node.right = build(current, j+1, end, preorder, inorder);
		return node;
	}
}
```



### 从上到下打印二叉树

层序遍历，结果以一维数组存储

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public int[] levelOrder(TreeNode root) {
        if (root == null)
            return new int[0];

        Queue<TreeNode> queue = new LinkedList<>();
        List<Integer> res = new ArrayList<>();
        queue.add(root);
        while (!queue.isEmpty()){
            int len = queue.size();
            for (int i = 0;i<len;i++){
                TreeNode tmp = queue.poll();
                if (tmp.left != null)
                    queue.add(tmp.left);
                if (tmp.right != null)
                    queue.add(tmp.right);
                res.add(tmp.val);
            }
        }
        int[] nums = new int[ res.size() ];
        for (int i = 0;i<res.size();i++){
            nums[i] = res.get(i);
        }
        return nums;
    }
}
```



### 从上到下打印二叉树Ⅲ

锯齿状遍历

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
    	List<List<Integer>> list = new ArrayList<>();
        if (root == null) {
        	return list;
        }
        //栈1来存储右节点到左节点的顺序
        Stack<TreeNode> stack1 = new Stack<>(); 
        //栈2来存储左节点到右节点的顺序
        Stack<TreeNode> stack2 = new Stack<>();
        
        //根节点入栈
    	stack1.push(root);
    	
    	//每次循环中，都是一个栈为空，一个栈不为空，结束的条件两个都为空
        while (!stack1.isEmpty() || !stack2.isEmpty()) {
        	List<Integer> subList = new ArrayList<>(); // 存储这一个层的数据
        	TreeNode cur = null;
        	
        	if (!stack1.isEmpty()) { //栈1不为空，则栈2此时为空，需要用栈2来存储从下一层从左到右的顺序
        		while (!stack1.isEmpty()) {	//遍历栈1中所有元素，即当前层的所有元素
        			cur = stack1.pop();
        			subList.add(cur.val);	//存储当前层所有元素
        			
        			if (cur.left != null) {	//左节点不为空加入下一层
        				stack2.push(cur.left);
        			}
        			if (cur.right != null) {	//右节点不为空加入下一层
        				stack2.push(cur.right);
        			}
        		}
        		list.add(subList);
        	}else {//栈2不为空，则栈1此时为空，需要用栈1来存储从下一层从右到左的顺序
        		while (!stack2.isEmpty()) {
        			cur = stack2.pop();
        			subList.add(cur.val);
        			
        			if (cur.right != null) {//右节点不为空加入下一层
        				stack1.push(cur.right);
        			}
        			if (cur.left != null) { //左节点不为空加入下一层
        				stack1.push(cur.left);
        			}
        		}
        		list.add(subList);
        	}
        }
        return list;
    }
}
```



### 字符串的排列

无重复元素的全排列

```java
class Solution {
    private LinkedList<String> list = new LinkedList<>();
    private StringBuilder sb;
    private char[] ch_array;
    private boolean[] selected;
    private int len;

    public String[] permutation(String s) {
        ch_array = s.toCharArray();
        //升序排序后，相同的char将会变得相邻
        Arrays.sort(ch_array);
        len = ch_array.length;
        selected = new boolean[len];
        sb = new StringBuilder(len);
        dfs(0);
        return list.toArray(String[]::new);
    }

    private void dfs(int order) {
        if (order == len) {
            list.addLast(sb.toString());
            return;
        }
        for (int i = 0; i < len; ++i) {
            //若当前字符已选，跳过
            if (selected[i] ||
                    //若上1个字符已选 且 当前字符 与 上1个字符相同，跳过
                    (i > 0 && selected[i - 1] && ch_array[i] == ch_array[i - 1]))
                continue;
            //加入当前字符
            sb.append(ch_array[i]);
            selected[i] = true;

            dfs(order + 1);
            //回溯到 未加入当前字符 的样子
            sb.deleteCharAt(order);
            selected[i] = false;
        }
    }
}
```



### 二叉搜索树与双向链表

```java
/*
class Node {
    public int val;
    public Node left;
    public Node right;

    public Node() {}

    public Node(int _val) {
        val = _val;
    }

    public Node(int _val,Node _left,Node _right) {
        val = _val;
        left = _left;
        right = _right;
    }
};
*/
class Solution {
    Node head=null,pre=null,tail=null;
    public Node treeToDoublyList(Node root) {
        if(root==null) return root;
        inorder(root);
        head.left=tail;
        tail.right=head;
        return head;
    }
    private void inorder(Node root){
        if(root==null) return ;
        inorder(root.left);
        if(pre==null) head=root;
        else pre.right=root;
        root.left=pre;
        pre=root;
        tail=root;
        inorder(root.right);
        return ;
    }
}
```



### 队列的最大值

```java
public class MaxQueue {
    Queue<Integer> queue;
    Deque<Integer> maxQueue;
    public MaxQueue() {
        queue=new ArrayDeque();
        maxQueue=new ArrayDeque();
    }
    public int max_value() {
        if(maxQueue.isEmpty())
            return -1;
        return maxQueue.peek();
    }
    public void push_back(int value) {
        queue.add(value);
        while(!maxQueue.isEmpty() && value>maxQueue.getLast())
            maxQueue.pollLast();
        maxQueue.add(value);
    }
    public int pop_front() {
        if(queue.isEmpty())
            return -1;
        int ans=queue.poll();
        if(ans==maxQueue.peek())
            maxQueue.poll();
        return ans;
    }
}

/**
 * Your MaxQueue object will be instantiated and called as such:
 * MaxQueue obj = new MaxQueue();
 * int param_1 = obj.max_value();
 * obj.push_back(value);
 * int param_3 = obj.pop_front();
 */
```



### 机器人的活动范围

```java
class Solution {
    public int movingCount(int m, int n, int k) {
        //BFS
        /*Queue<int[]> q = new LinkedList<int[]>();
        q.add(new int[]{0,0});
        int[][] visited = new int[m][n];
        int count = 0;
        while(!q.isEmpty()){
            int[] point = q.poll();
            int x = point[0];
            int y = point[1];
            if(visited[x][y] == 0){
                visited[x][y] = 1;
                int sum_x = 0, sum_y = 0, sum = 0;
                sum_x = x / 10 + x % 10;
                sum_y = y / 10 + y % 10;
                sum = sum_x + sum_y;
                if(sum <= k){
                    count++;
                    if(y + 1 < n && visited[x][y + 1] == 0)
                        q.add(new int[]{x, y + 1});
                    if(x + 1 < m && visited[x + 1][y] == 0)
                        q.add(new int[]{x + 1, y});
                    if(y - 1 >= 0 && visited[x][y - 1] == 0)
                        q.add(new int[]{x, y - 1});
                    if(x - 1 >= 0 && visited[x - 1][y] == 0)
                        q.add(new int[]{x - 1, y});
                }
            }
        }
        return count;*/
        //DFS
        Stack<int[]> s = new Stack<int[]>();
        s.push(new int[]{0,0});
        int[][] visited = new int[m][n];
        int count = 0;
        while(!s.isEmpty()){
            int[] point = s.pop();
            int x = point[0];
            int y = point[1];
            if(visited[x][y] == 0){
                visited[x][y] = 1;
                int sum_x = 0, sum_y = 0, sum = 0;
                sum_x = x / 10 + x % 10;
                sum_y = y / 10 + y % 10;
                sum = sum_x + sum_y;
                if(sum <= k){
                    count++;
                    if(y + 1 < n && visited[x][y + 1] == 0)
                        s.push(new int[]{x, y + 1});
                    if(x + 1 < m && visited[x + 1][y] == 0)
                        s.push(new int[]{x + 1, y});
                    if(y - 1 >= 0 && visited[x][y - 1] == 0)
                        s.push(new int[]{x, y - 1});
                    if(x - 1 >= 0 && visited[x - 1][y] == 0)
                        s.push(new int[]{x - 1, y});
                }
            }
        }
        return count;
    }
}
```



### 1-n整数中1出现的次数

递归

```java
class Solution{
    private int dfs(int n) {
        if (n <= 0) {
            return 0;
        }

        String numStr = String.valueOf(n);
        int high = numStr.charAt(0) - '0';
        int pow = (int) Math.pow(10, numStr.length() - 1);
        int last = n - high * pow;

        if (high == 1) {
            // 最高位是1，如1234, 此时pow = 1000,那么结果由以下三部分构成：
            // (1) dfs(pow - 1)代表[0,999]中1的个数;
            // (2) dfs(last)代表234中1出现的个数;
            // (3) last+1代表固定高位1有多少种情况。
            return dfs(pow - 1) + dfs(last) + last + 1;
        } else {
            // 最高位不为1，如2234，那么结果也分成以下三部分构成：
            // (1) pow代表固定高位1，有多少种情况;
            // (2) high * dfs(pow - 1)代表999以内和1999以内低三位1出现的个数;
            // (3) dfs(last)同上。
            return pow + high * dfs(pow - 1) + dfs(last);
        }
    }

    // 递归求解
    public int countDigitOne(int n) {
        return dfs(n);
    }
}
```



### 树的子结构

```java
class Solution {
    public boolean isSubStructure(TreeNode A, TreeNode B) {
        return (A != null && B != null) && (recur(A, B) || isSubStructure(A.left, B) || isSubStructure(A.right, B));
    }
    boolean recur(TreeNode A, TreeNode B) {
        if(B == null) return true;
        if(A == null || A.val != B.val) return false;
        return recur(A.left, B.left) && recur(A.right, B.right);
    }
}
```



### 把字符串转换成整数

```java
class Solution {
    /**
     * mark :  表示正负号
     * flag : 表示是否开始处理 数字、+号
     * pop : 表示进位
     * @param str
     * @return
     */
    public int strToInt(String str) {
        if(str == null || str.length() == 0) return 0;
        //标记正负号
        int mark = 0;
        //标志取到的第一位是否是数字
        boolean flag = false;
        //存放最终输出的数值
        int number = 0;
        String newStr = str.trim();
        char[] chars = newStr.toCharArray();
        //判断是否为空
        if(chars.length == 0) return 0;
        //若为负号
        if(chars[0] == '-') mark = 1;
        for(int i = mark;i < chars.length;i++){
            //已经判定为负号，却又遇到正号
            if (mark == 1 && chars[i] == '+' && flag == false) {
                return 0;
            }else if (mark == 1 && chars[i] == '+' && flag == true){
                break;
            }
            if(mark == 0 && chars[i] == '+' && flag == false){
                flag = true;
                //第一次遇到正号
                continue;
            }else if (mark == 0 && chars[i] == '+' && flag == true) {
                //第二次遇到正号
                break;
            }
            //判定
            if((chars[i] < '0' || chars[i] > '9') && flag == false){
                //当第一位为非0 - 9
                return 0;
            }else if((chars[i] < '0' || chars[i] > '9') && flag == true){
                break;
            }else{
                int pop = chars[i] - '0';
                //进行最大值的判断与溢出判断
                if ((number > Integer.MAX_VALUE/10 ||
                        (number == Integer.MAX_VALUE/10 && pop > Integer.MAX_VALUE % 10)) && mark == 0)
                    return Integer.MAX_VALUE;
                if ((-number < Integer.MIN_VALUE/10 ||
                        (-number == Integer.MIN_VALUE/10 && -pop < Integer.MIN_VALUE % 10)) && mark == 1)
                    return Integer.MIN_VALUE;
                //char[i]-'0'为转成数字
                number = number * 10 + pop;
                flag = true;
            }
        }
        return mark != 1?number:-number;
    }
}
```

