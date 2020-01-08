package perceptron;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;
import mnisttools.MnistReader;
import java.util.Arrays;



public class QuestionSeven {
    /* Les donnees */
    public static String path=perceptronMulti.path;
    public static String labelDB=perceptronMulti.labelDB;
    public static String imageDB=perceptronMulti.imageDB;

    /* Parametres */
    // Na exemples pour l'ensemble d'apprentissage
    public static final int Na = 2000;
    // Nv exemples pour l'ensemble d'évaluation
    public static final int Nv = 1000;
    // Nombre d'epoque max
    public final static int EPOCHMAX=500;
    // Classe positive (le reste sera considere comme des ex. negatifs):
    public static int  classe = 26 ;
    public static float eta = (float) 0.007;
    // Générateur de nombres aléatoires
    public static int seed = 1234;
    public static Random GenRdm = new Random();

    public static MnistReader db = new MnistReader(labelDB, imageDB);
    /* Tableau où stocker les données */
    public static int longueur=db.getImage(1)[1].length;
    public static int largeur=db.getImage(1).length;

    // Initialisation des tableaux d'apprentissage et de validation
    public static float[][] trainData= new float[Na][largeur*longueur+1];
    public static float[][] validData= new float[Nv][largeur*longueur+1];
    public static int[] label = new int[Na];
    public static int[] labelVal = new int[Na];

    public static void main(String[] arg)  throws IOException {
        FileWriter fw = new FileWriter("questionSeven");
        int labelInt, j, o, epoc;
        // on rempli les tableaux de l'ensemble de validation à 1000
        o = 0;
        j = 1;
        while(j < Nv){
            o += 1;
            labelInt = db.getLabel(o);
            if (labelInt >= 10 && labelInt <= 21){
                labelVal[j]=labelInt;
                validData[j]= perceptronMulti.ConvertImage(perceptronMulti.BinariserImage(db.getImage(o),(int)255/2));
                j += 1;
            }
        }


        // on rempli les tableaux de l'ensemble d'apprentissage à 1000
        j = 1;
        while( j < Na + 1){
            o += 1;
            labelInt = db.getLabel(o);
            if (labelInt >= 10 && labelInt <= 21){
                label[j-1] = labelInt;
                trainData[j-1] = perceptronMulti.ConvertImage(perceptronMulti.BinariserImage(db.getImage(o),255/2));
                j += 1;
            }
        }


        int err=0,errVal=0;
        float[][] poids = perceptronMultiSeven.InitialiseW(largeur*longueur+1); //poids[class][donnée xi]
        float[][] result = new float[Na][classe]; //result[Na][class] récupere les probalités que Na est dans la classe

        float[][] resultVal = new float[Na][classe];

        int maxInd=0,maxIndVal=0;

        epoc = 0;
        do{
            err = 0;
            errVal=0;
            for(int i=0; i<Na;i++) { //i<Na
                result[i]=perceptronMultiSeven.InfPerceptron(trainData[i], poids); //Initialise les results
                perceptronMultiSeven.Majpoids(trainData[i], poids, eta, result[i], label[i]);

                if(i < Nv){
                    resultVal[i]=perceptronMultiSeven.InfPerceptron(validData[i], poids); //Initialise les results
                    maxIndVal = perceptronMultiSeven.MaxIndiceTab(resultVal[i]);
                    if(labelVal[i]-10!=maxIndVal) {
                        errVal+=1;
                    }
                }

                maxInd = perceptronMultiSeven.MaxIndiceTab(result[i]);

                if(label[i]-10!=maxInd) {
                    err+=1;
                }

            }
            System.out.println(epoc + " (" + err + "~" + errVal + ")");
            epoc+=1;

            fw.write(epoc+ " " +err+" "+errVal);
            fw.write("\n");


        } while(epoc<EPOCHMAX && err > 0);

        fw.close();

    }

}

