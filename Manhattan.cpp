#include<bits/stdc++.h>
#define faster ios_base::sync_with_stdio(0);cin.tie(0);cout.tie(0);
#define db double
#define bo bool
#define vo void
#define ch char
#define fl float
#define ll long long
#define ull unsigned long long
#define str string
#define re return
#define all(x) (x).begin() , (x).end()
#define allr(x) (x).rbegin() , (x).rend()
#define v(x) vector<x>
#define NAME "name"
using namespace std;
const ll INF=(1LL<<60);
int main()
{
    faster
    if(fopen(NAME".inp","r"))
	{
	    freopen(NAME".inp","r",stdin);
        freopen(NAME".out","w",stdout);
	}
	ll q,n,m,min1,min2,max1,max2,i,j,t,d;
	str s;
	cin>>q;
    while(q--)
	{
		cin>>n>>m;
		min1=INF;
		min2=min1;
		max1=-INF;
		max2=max1;
        for(i=1;i<=n;i++)
		{
            cin>>s;
            for(j=1;j<=m;j++)
			{
                if(s[j-1]=='#')
				{
                    t=i+j;
                	d=i-j;
                    if(t<min1) min1=t;
                    if(t>max1) max1=t;
                    if(d<min2) min2=d;
                    if(d>max2) max2=d;
                }
            }
        }
        cout<<(min1+min2+max1+max2)/4<<" "<<(min1+max1-min2-max2)/4<<"\n";
    }
	re 0;
}
