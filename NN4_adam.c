#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

int main(void)
{
/*ファイルの読み込み*/
FILE *fp;
fp=fopen("mnist_small.txt","r");
if(fp==NULL){
printf("file open error\n");
}
else{
printf("file opened\n");
}

/*データを配列に格納*/
float num[400][785];
int nu;
for(int j=0;j<400;j++){
    for(int i=0;i<785;i++){
        fscanf(fp,"%d%*[^,]",&nu); /*カンマで区切られているのでカンマを取り除く*/
        if(i==0){
            num[j][i]=nu;
        }
        if(i!=0){
        num[j][i]=nu/255.0;  /*0~255の値をとるので、255で割って0～1の範囲に変換する*/
        }
    }
}
fclose(fp);

/*各種変数の定義*/
int N=1;  /*学習の繰り返し回数*/
float rate=0.1;  /*学習率*/
int M=100;  /*隠れ層の厚さ*/
float W1[784][M];  /*一層目の重み*/
float a1[M][200];  /*一層目の出力*/
float z1[M][200];  /*一層目の出力を活性化関数に入力したもの*/
float W2[M][10];  /*二層目の重み*/
float a2[10][200];  /*二層目の出力*/
float z2[10][200];  /*二層目の出力をSOFTMAX関数に入力したもの*/
float b1=0.1;  /*一層目のバイアス*/
float b2=0.1;  /*二層目のバイアス*/


/*重みをランダムな数で初期化する*/
for(int i=0;i<784;i++){
    for(int j=0;j<M;j++){
        W1[i][j]=(double)rand()/327670;
    }
}
for(int i=0;i<M;i++){
    for(int j=0;j<10;j++){
        W2[i][j]=(double)rand()/327670;
    }
}

/*正解データをone-hotベクトル化する*/
float ans[10][400];
for(int j=0;j<400;j++){
    if(fabsf(num[j][0]-0.0)<1){
       ans[0][j]=1.0;
    }
    if(fabsf(num[j][0]-1.0)<1){
        ans[1][j]=1.0;
    }
    if(fabsf(num[j][0]-2.0)<1){
        ans[2][j]=1.0;
    }
    if(fabsf(num[j][0]-3.0)<1){
        ans[3][j]=1.0;
    }
    if(fabsf(num[j][0]-4.0)<1){
        ans[4][j]=1.0;
    }
    if(fabsf(num[j][0]-5.0)<1){
        ans[5][j]=1.0;
    }
    if(fabsf(num[j][0]-6.0)<1){
       ans[6][j]=1.0;
    }
    if(fabsf(num[j][0]-7.0)<1){
        ans[7][j]=1.0;
    }
    if(fabsf(num[j][0]-8.0)<1){
        ans[8][j]=1.0;
    }
    if(fabsf(num[j][0]-9.0)<1){
        ans[9][j]=1.0;
    }
    
}

/*ここから順伝播と逆伝播を実装していく*/
for(int o=0;o<N;o++){

/*ここからは順伝播の実装*/
for(int n=0;n<200;n++){
    for(int m=0;m<M;m++){
        for(int l=0;l<784;l++){
            a1[m][n]=a1[m][n]+num[n][l+1]*W1[l][m]/784+b1/784;  /*一層目の計算*/
        }
    }
    
    for(int i=0;i<M;i++){
        z1[i][n]=(0.55*a1[n][i]+0.45*fabsf(a1[n][i]));  /*活性化関数Leaky ReLUを適用する*/
        }

    for(int m=0;m<10;m++){
        for(int l=0;l<M;l++){
            a2[m][n]=a2[m][n]+z1[l][n]*W2[l][m]/M+b2/M;  /*二層目の計算*/
        }
    }
    
    /*ここからSOFTMAX関数の実装*/
    float u=0.0;
    float t=0.001;
    for(int i=0;i<10;i++){
        if(t<a2[i][n]){
            t=a2[i][n];
        }
    }
    for(int i=0;i<10;i++){
        u=u+(exp(a2[i][n]-t));  /*数値がオーバーフローしないように最大値tのexpで割る*/
    } 
    for(int i=0;i<10;i++){
        z2[i][n] = exp(a2[i][n]-t)/(u+0.00001);  /*数値がオーバーフローしないように最大値tのexpで割る*/
    }
    
    /*ここから逆伝播の実装*/
    for(int i=0;i<M;i++){
        for(int j=0;j<10;j++){
        float pre=z2[j][n]-ans[j][n]*rate;
        W2[i][j]=W2[i][j]-(z1[i][n]*pre)*0.55-fabsf(z1[i][n]*pre)*0.45;  /*二層目の重みの更新*/
        }
    }

    for(int j=0;j<784;j++){
        for(int i=0;i<M;i++){
            float pre[M];
            for(int l=0;l<10;l++){
            W1[j][i]=W1[j][i]-num[n][j]*((z2[l][n]-ans[l][n])*W2[i][l]*rate*0.55+fabsf((z2[l][n]-ans[l][n])*W2[i][l]*rate)*0.45);  /*一層目の重みの更新*/
            }
        }
    }
}

/*ここからは更新したパラメータを用いて順伝播を計算し、誤差と正解率を求める*/
for(int n=0;n<200;n++){
    for(int m=0;m<M;m++){
        for(int l=0;l<784;l++){
            a1[m][n]=a1[m][n]+num[n+200][l+1]*W1[l][m]/784+b1/784;
        }
    }
    
    for(int i=0;i<M;i++){
        z1[i][n]=(0.55*a1[n][i]+0.45*fabs(a1[n][i]));
    }


    for(int m=0;m<10;m++){
        for(int l=0;l<M;l++){
            a2[m][n]=a2[m][n]+z1[l][n]*W2[l][m]/M+b2/M;
        }
    }
    


    float u=0.0;
    float t=0.001;
    for(int i=0;i<10;i++){
        if(t<a2[i][n]){
            t=a2[i][n];
        }
    }
    for(int i=0;i<10;i++){
        u=u+(exp(a2[i][n]-t));
    } 
    for(int i=0;i<10;i++){
        z2[i][n] = exp(a2[i][n]-t)/(u+0.00001);
    }
}

/*交差エントロピー誤差を求める*/
float score=0.0;
for(int i=0;i<200;i++){
    for(int j=0;j<10;j++){
    score=score-ans[j][i+200]*log(z2[j][i]+0.000001)/200;
    }
}

/*正解率を求める*/
float acc=0.0;
for(int j=0;j<200;j++){
        float z=z2[0][j];
        float zindex=0.0;
        if(z<z2[1][j]){
            z=z2[1][j];
            zindex=1.0;
        }
        if(z<z2[2][j]){
            z=z2[2][j];
            zindex=2.0;
        }
        if(z<z2[3][j]){
            z=z2[3][j];
            zindex=3.0;
        }
        if(z<z2[4][j]){
            z=z2[4][j];
            zindex=4.0;
        }
        if(z<z2[5][j]){
            z=z2[5][j];
            zindex=5.0;
        }
        if(z<z2[6][j]){
            z=z2[6][j];
            zindex=6.0;
        }
        if(z<z2[7][j]){
            z=z2[7][j];
            zindex=7.0;
        }
        if(z<z2[8][j]){
            z=z2[8][j];
            zindex=8.0;
        }
        if(z<z2[9][j]){
            z=z2[9][j];
            zindex=9.0;
        }
        if(fabsf(zindex-num[j+200][0])<1){
            acc=acc+0.5;
        }
}

/*結果の表示*/
printf("%d回目計算終了\n",o+1);
printf("error : %lf\n",score);
printf("accuracy : %lf\n",acc);

}
return 0;
}


