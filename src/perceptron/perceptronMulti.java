package perceptron;

import java.util.Random;

import mnisttools.MnistReader;

import java.lang.Math;

public class perceptronMulti {
   /* Les donnees */
    public static String path="/home/jerome/downloads/";
    public static String imageDB=path+"emnist-byclass-train-images-idx3-ubyte";
    public static String labelDB=path + "emnist-byclass-train-labels-idx1-ubyte";


    public static int  classe = 21 - 10 + 1 ;
    public static int  minlabel = 10 ;
    public static Random GenRdm = new Random();

    /*
     *  BinariserImage :
     *      image: une image int à deux dimensions (extraite de MNIST)
     *      seuil: parametre pour la binarisation
     *
     *  on binarise l'image à l'aide du seuil indiqué
     *
     */
    public static int[][] BinariserImage(int[][] image, int seuil) {
        int pixel;
        for(int i = 0 ; i < image.length ; i++) {
            for(int j = 0 ; j < image[0].length ; j++) {
                pixel = image[i][j];
                if(pixel >= seuil) {
                    image[i][j] = 1;
                }

                else {
                    image[i][j]=0;
                }
            }
        }
        return image;
    }

    /*
     *  ConvertImage :
     *      image: une image int binarisée à deux dimensions
     *
     *  1. on convertit l'image en deux dimension dx X dy, en un tableau unidimensionnel de tail dx.dy
     *  2. on rajoute un élément en première position du tableau qui sera à 1
     *  La taille finale renvoyée sera dx.dy + 1
     *
     */
    public static float[] ConvertImage(int[][] image) {
        int dim = image.length * image[0].length + 1;
        float convertedImage[];
        convertedImage = new float[dim];
        convertedImage[0] = 1;
        int k = 1;
        for(int i = 0 ; i < image.length ; i++){
            for(int j = 0 ; j < image[0].length ; j++){
                convertedImage[k] = image[i][j];
                k += 1;
            }
        }
        return convertedImage;
    }


    //Créer un vecteur ei de taille classe=10
    public static int[] OneHot(int label) {
        int[] result = new int[classe];
        for(int i=0 ;i<result.length;i++) {
            result[i]=0;
        }
        try{

            result[label-minlabel]=1;
        }catch(Exception e){
            System.out.print("Error label");
        }
        return result;
    }

    public static void Majpoids(float[] donnee,float poids[][],float eta, float[] result,int label){
        for(int l=0; l<poids.length;l++) { //classe l
            for(int i=0; i<poids[l].length;i++) { //nb de donnée
                poids[l][i] -= donnee[i]*eta*(result[l]-OneHot(label)[l]);
            }
        }
    }

    // sum des donnee[i]*poids[i] taille (largeur*longueur+1)
    public static float sumDonnee(float[] donnee, float[] poids) {
        float sum=0;
        for(int i=0; i<poids.length; i++) {
            sum+= donnee[i]*poids[i];
        }
        return sum;
    }


    // return le tableau qui contient la probabilité que l'image soit dans la classe l (taille classe)
    public static float[] InfPerceptron(float[] donnee, float[][] poids) {
        float[] result = new float[classe];
        float den=0;
        for(int i=0; i<poids.length; i++) {
            den+=Math.exp(sumDonnee(donnee,poids[i]));
        }

        for(int i=0; i<result.length;i++) {
            result[i] = (float) Math.exp(sumDonnee(donnee,poids[i]))/den;
            //System.out.println(result[i]);
        }
        return result;
    }


    public static float[][] InitialiseW(int sizeW) {
        float[][] result = new float[classe][sizeW];
        for(int i=0; i<classe; i++)
            for(int j=0;j<sizeW;j++) {
                result[i][j]=(float)1/(float)sizeW*(float)GenRdm.nextInt(101)/(float)100;
            }
        return result;
    }

    public static int MaxIndiceTab(float[] tab){
        float max = tab[0];
        int maxInd = 0;
        for(int i = 0; i < tab.length ; i++ ){
            if(tab[i] > max){
                max = tab[i];
                maxInd = i;
                }
        }
        return maxInd;
    }
    
    public static float MaxTab(float[] tab){
        float max = tab[0];
        for(int i = 0; i < tab.length ; i++ ){
            if(tab[i] > max){
                max = tab[i];
                }
        }
        return max;
    }
    public static int MinIndiceTab(float[] tab){
        float min = tab[0];
        int minInd = 0;
        for(int i = 0; i < tab.length ; i++ ){
            if(tab[i] < min){
                min = tab[i];
                minInd = i;
                }
        }
        return minInd;
    }
    

    public static float MinTab(float[] tab){
        float min = tab[0];
        for(int i = 0; i < tab.length ; i++ ){
            if(tab[i] < min){
                min = tab[i];
                }
        }
        return min;
    }
    
    public static void MatConfusion(float[][] result, int[] label ,int[][] Matresult){
    	int maxInd;
    	for(int i=0; i<Matresult.length; i++) {
    		for(int j=0; j<Matresult[i].length; j++) {
    			Matresult[i][j]=0;
    		}
    	}
    	
    	for(int i=0; i<label.length;i++) {
    		maxInd = MaxIndiceTab(result[i]);
    		//System.out.println(label[i]-10+" "+ maxInd);

    		Matresult[label[i]-10][maxInd]+=1;
    	}
    }
  
    
}
