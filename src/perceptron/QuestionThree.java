package perceptron;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

import mnisttools.MnistReader;

public class QuestionThree {
    /* Les donnees */
    public static String path=perceptronMulti.path;
    public static String labelDB=perceptronMulti.labelDB;
    public static String imageDB=perceptronMulti.imageDB;

    /* Parametres */
    // Na exemples pour l'ensemble d'apprentissage
    public static final int Na = 5000;
    // Nv exemples pour l'ensemble d'évaluation
    public static final int Nv = 1000;
    // Nombre d'epoque max
    public final static int EPOCHMAX=50;
    // Classe positive (le reste sera considere comme des ex. negatifs):
    public static int  classe = 12 ;
    public static float eta = (float) 0.05;
    // Générateur de nombres aléatoires
    public static int seed = 1234;
    public static Random GenRdm = new Random();

    public static MnistReader db = new MnistReader(labelDB, imageDB);
    /* Tableau où stocker les données */
    public static int longueur=db.getImage(1)[1].length;
    public static int largeur=db.getImage(1).length;

    /* Donnée d'apprentissage */
    public static float[][] trainData= new float[Na][largeur*longueur+1];
    public static float[][] validData= new float[Nv][largeur*longueur+1];
    public static int[] label = new int[Na];
    public static int[] labelVal = new int[Nv];

    public static void main(String[] arg)  throws IOException {
        FileWriter fw = new FileWriter("questionThree.MatApp.d");
        FileWriter fx = new FileWriter("questionThree.MatVal.d");

        int labelInt, j, o, epoc = 0;
        o = 0;
        j = 0;
        while( j < Na ){
            o += 1;
            labelInt = db.getLabel(o);
            if (labelInt >= 10 && labelInt <= 21){
                label[j] = labelInt;
                trainData[j] = perceptronMulti.ConvertImage(perceptronMulti.BinariserImage(db.getImage(o),(int)255/2));
                j += 1;
            }
        }

        o = 697932;
        j = 0;
        // on rempli l'ensemble de validation en partant de la fin de la base de donnée
        while(j < Nv){
            o -= 1;
            labelInt = db.getLabel(o);
            if (labelInt >= 10 && labelInt <= 21){
                labelVal[j]=labelInt;
                validData[j]= perceptronMulti.ConvertImage(perceptronMulti.BinariserImage(db.getImage(o),(int)255/2));
                j += 1;
            }
        }

        float[][] poids = perceptronMulti.InitialiseW(largeur*longueur+1); //poids[class][donnée xi]
        float[][] result = new float[Na][classe]; //result[Na][class] récupere les probalités que Na est dans la classe

        float[][] resultVal = new float[Nv][classe];

        
        int[][] matriceConfusion = new int[classe][classe];
        int[][] matriceConfusionVal = new int[classe][classe];
        char[] Caractere = new char[classe];
        
        int err=0, errval=0;
        int maxind=0, maxindval=0;
        do{
            for(int i=0; i<Na;i++) { //i<Na
                result[i]=perceptronMulti.InfPerceptron(trainData[i], poids); //Initialise les results
                perceptronMulti.Majpoids(trainData[i], poids, eta, result[i], label[i]);
                maxind = perceptronMulti.MaxIndiceTab(result[i]);
                if(maxind != label[i]-10) {
                	err+=1;
                }

            }
            epoc+=1;
            System.out.println(err);
            err=0;

        } while(epoc<EPOCHMAX);
        
        for(int i=0; i<Nv; i++) {
            resultVal[i]=perceptronMulti.InfPerceptron(validData[i], poids); //Initialise les results
            maxindval = perceptronMulti.MaxIndiceTab(resultVal[i]);
            if(maxindval != labelVal[i]-10) {
            	errval+=1;
            }

        }
        
        System.out.println(errval);

        perceptronMulti.MatConfusion(result, label, matriceConfusion);
        perceptronMulti.MatConfusion(resultVal, labelVal, matriceConfusionVal);
        String A = "ABCDEFGHIJKL";
        Caractere=A.toCharArray();
        
        fw.write(" ");
    	fx.write(" ");

        for(int i=0; i<Caractere.length;i++) {
        	fw.write(Caractere[i]+ " ");
        	fx.write(Caractere[i]+ " ");
        }
        
        fw.write("\n");
    	fx.write("\n");

        for(int i=0; i<matriceConfusion.length; i++){
        	fw.write(Caractere[i]+ " ");
        	fx.write(Caractere[i]+ " ");
        	for(j=0; j<matriceConfusion[i].length; j++) {
        		fw.write(matriceConfusion[i][j]+ " ");
        		fx.write(matriceConfusionVal[i][j]+ " ");

        	}
            fw.write("\n");
            fx.write("\n");

        }
        fw.close();
        fx.close();


    }


}
