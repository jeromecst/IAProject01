package perceptron;

import java.util.Random;
import mnisttools.MnistReader;
import java.util.Arrays;
import java.io.FileWriter;
import java.io.IOException;



public class QuestionOne {
    /* Les donnees */
    public static String path=perceptronMulti.path;
    public static String labelDB=perceptronMulti.labelDB;
    public static String imageDB=perceptronMulti.imageDB;

 /* Parametres */
    // Nv exemples pour l'ensemble d'évaluation
    public static final int Nv = 1000;
    // Nombre d'epoque max
    public final static int EPOCHMAX=40;
    // Classe positive (le reste sera considere comme des ex. negatifs):
    public static int  classe = 12 ;
    public static float eta = (float) .05;
    // Générateur de nombres aléatoires
    public static int seed = 1234;
    public static Random GenRdm = new Random();

    public static MnistReader db = new MnistReader(labelDB, imageDB);
    /* Tableau où stocker les données */
    public static int longueur=db.getImage(1)[1].length;
    public static int largeur=db.getImage(1).length;

    /* Donnée d'apprentissage */

    public static int NaMax = 10000; //on fixe NaMax à 10000 puis on travaillera sur Na entre 1000 et NaMax


    public static void main(String[] arg)  throws IOException {
        float[][] trainData= new float[NaMax][largeur*longueur+1];
        float[][] validData= new float[NaMax][largeur*longueur+1];
        int[] label = new int[NaMax];
        int[] labelVal = new int[NaMax];
        int labelInt, j=0, o=697932, epoc, minErrVal;

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

        // on rempli les tableaux de l'ensemble d'apprentissage à 10000
        j = 1;
        while( j < NaMax + 1){
            o += 1;
            labelInt = db.getLabel(o);
            if (labelInt >= 10 && labelInt <= 21){
                label[j-1] = labelInt;
                trainData[j-1] = perceptronMulti.ConvertImage(perceptronMulti.BinariserImage(db.getImage(o),255/2));
                j += 1;
            }
        }

        float [][] poids = perceptronMulti.InitialiseW(largeur*longueur+1); //poids[class][donnée xi]
        float[][] result = new float[NaMax][classe]; //result[Na][class] récupere les probalités que Na est dans la classe
        float[][] resultVal = new float[NaMax][classe];


        FileWriter fw = new FileWriter("questionOne");

        for(int Na = 1000; Na < 10000 ; Na += 100){
            int err=0,errVal=0;
            poids = perceptronMulti.InitialiseW(largeur*longueur+1);
            int maxInd=0,maxIndVal=0;

            epoc = 0;
            minErrVal=1000;
            do {
                err = 0;
                errVal = 0;
                for (int i = 0; i < Na; i++) { //i<Na
                    result[i] = perceptronMulti.InfPerceptron(trainData[i], poids); //Initialise les results
                    perceptronMulti.Majpoids(trainData[i], poids, eta, result[i], label[i]);
                    maxInd = perceptronMulti.MaxIndiceTab(result[i]);

                    if(i < Nv){
                        resultVal[i] = perceptronMulti.InfPerceptron(validData[i], poids); //Initialise les results
                        maxIndVal = perceptronMulti.MaxIndiceTab(resultVal[i]);
                        if (labelVal[i] - 10 != maxIndVal ) {
                            errVal += 1;
                        }
                    }

                    if (label[i] - 10 != maxInd) {
                        err += 1;
                    }


                }
                System.out.println(epoc + " (" + err + "~" + errVal + ") " + Na);
                epoc += 1;

                if(errVal < minErrVal){
                    minErrVal = errVal;
                }

            } while (epoc < EPOCHMAX && err > (int)(.005*Na));
            // à moins de 5% d'erreurs, on stoppe l'aglorithme

            fw.write(Na+" "+minErrVal);
            fw.write("\n");
        }
        fw.close();
    }
}
