package fosalgo;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Scanner;

public class Perceptron {

    private final File fileModel = new File("model.txt");//file model

    public String training(File fileDataTraining, double alpha, double theta) {
        String result = null;
        StringBuffer sb = new StringBuffer();
        //Baca dataset  
        ArrayList<Data> dataset = null;
        HashSet<Character> setHuruf = new HashSet<>();//deteksi huruf unik
        try {
            File file = fileDataTraining;
            Scanner sc = new Scanner(file);
            dataset = new ArrayList<>();
            while (sc.hasNextLine()) {
                String baris = sc.nextLine();
                String[] kolom = baris.split(":");
                char huruf = kolom[0].charAt(0);
                String[] sPola = kolom[1].split(";");
                //konversi sPola ke array integer dengan menambahkan konstan 1 di elemen pertama
                int[] pola = new int[sPola.length + 1];
                pola[0] = 1;//konstan 1 akan berpasangan dengan bias
                int k = 1;
                for (int i = 0; i < sPola.length; i++) {
                    pola[k] = Integer.parseInt(sPola[i]);
                    k++;
                }
                Data data = new Data(huruf, pola);
                dataset.add(data);
                setHuruf.add(huruf);
            }

        } catch (FileNotFoundException ex) {
            ex.printStackTrace();
        }//end of baca dataset dari file

        //size
        int inSize = dataset.get(0).pola.length;//n, menyatakan banyaknya input
        int outSize = setHuruf.size();//m, menyatakan banyaknya output
        int dataSize = dataset.size();//menyatakan banyaknya data training

        //encode target        
        HashMap<Character, int[]> encodeTarget = new HashMap<>();
        int k = 0;
        for (char c : setHuruf) {
            int[] target = new int[outSize];
            for (int i = 0; i < outSize; i++) {
                target[i] = -1;
            }
            target[k] = 1;
            encodeTarget.put(c, target);
            k++;
        }

        //set target untuk semua elemen dataset
        for (Data d : dataset) {
            char huruf = d.huruf;
            int[] target = encodeTarget.get(huruf);
            d.target = target;
        }

        //print dataset
        //System.out.println(dataset);
        //TRAINING PERCEPTRON---------------------------------------------------
        double[][] W = new double[inSize][outSize];//W = weights = model yang akan dilatih
        int epoch = 0;
        int MAX_EPOCH = 1000;
        while (true) {
            epoch++;
            //System.out.println("EPOCH: " + epoch);
            int nSama = 0;
            for (Data d : dataset) {
                int[] X = d.pola;
                int[] target = d.target;

                //SUMMATION
                //Hitung net
                double[] net = new double[outSize];
                for (int i = 0; i < inSize; i++) {
                    int Xi = X[i];
                    for (int j = 0; j < outSize; j++) {
                        double Wij = W[i][j];
                        double XiWij = Xi * Wij;
                        net[j] += XiWij;
                    }
                }//hitung net selesai

                //ACTIVATION
                int[] y = new int[outSize];
                for (int j = 0; j < outSize; j++) {
                    if (net[j] > theta) {
                        y[j] = 1;
                    } else if (net[j] >= -theta) {
                        y[j] = 0;
                    } else {
                        y[j] = -1;
                    }
                }//hitung aktifasi selesai

                //CEK UPDATE BOBOT
                double[][] deltaW = new double[inSize][outSize];//inisialisasi delta W
                boolean equals = true;
                for (int j = 0; j < outSize; j++) {
                    if (target[j] != y[j]) {
                        //hitung delta W
                        for (int i = 0; i < inSize; i++) {
                            deltaW[i][j] = alpha * target[j] * X[i];
                        }
                        equals = false;
                    }
                }

                if (equals) {
                    nSama++;
                }

                //UPDATE BOBOT
                for (int i = 0; i < inSize; i++) {
                    for (int j = 0; j < outSize; j++) {
                        W[i][j] += deltaW[i][j];
                    }
                }
            }//end of for (Data d : dataset)

            //CEK Kondisi Berhenti
            if (nSama == dataSize || epoch >= MAX_EPOCH) {
                //PROSES TRAINING SELESAI
                //System.out.println("TRAINING SELESAI");
                break;
            }

        }//end of while
        //PROSES TRAINING SELESAI-----------------------------------------------

        //SAVE HASIL TRAINING
        //Save Model / Bobot / Weights
        sb.append("input size:" + inSize + "\n");
        sb.append("output size:" + outSize + "\n");
        sb.append("theta:" + theta + "\n");
        sb.append("CHARACTER:\n");
        for (Character c : encodeTarget.keySet()) {
            int[] target = encodeTarget.get(c);
            sb.append(c + ":");
            for (int j = 0; j < target.length; j++) {
                if (j > 0) {
                    sb.append(";");
                }
                sb.append(target[j]);
            }
            sb.append("\n");
        }
        sb.append("WEIGHTS:\n");
        for (int i = 0; i < inSize; i++) {
            for (int j = 0; j < outSize; j++) {
                if (j > 0) {
                    sb.append(";");
                }
                sb.append(String.format("%.6f", W[i][j]));//simpan dalam format enam tempat decimal
            }
            sb.append("\n");
        }

        //write to file  
        try {
            Path path = fileModel.toPath();
            Files.writeString(path, sb.toString());
        } catch (IOException ex) {
            ex.printStackTrace();
        }
        result = sb.toString();
        return result;
    }//end of training

    public String testing(File fileTesting) {
        String result = null;
        StringBuffer sb = new StringBuffer();
        double theta = 0;
        int inSize = 0;
        int outSize = 0;

        HashMap<Character, int[]> encodeTarget = null;//encode target 
        double[][] W = null;//Model / Bobot / Weight        
        ArrayList<Data> dataTesting = null;

        //Baca model dari file    
        try {
            File file = fileModel;
            Scanner sc = new Scanner(file);
            String baris = sc.nextLine();
            inSize = Integer.parseInt(baris.split(":")[1]);//size of input 

            baris = sc.nextLine();
            outSize = Integer.parseInt(baris.split(":")[1]);//size of output

            baris = sc.nextLine();
            theta = Double.parseDouble(baris.split(":")[1]);

            sc.nextLine();//skip line CHARACTER:

            //encode target       
            encodeTarget = new HashMap<>();//encode target 
            for (int j = 0; j < outSize; j++) {
                baris = sc.nextLine();
                String[] kolom = baris.split(":");
                char huruf = kolom[0].charAt(0);
                String[] sTarget = kolom[1].split(";");
                int[] target = new int[sTarget.length];
                for (int k = 0; k < sTarget.length; k++) {
                    target[k] = Integer.parseInt(sTarget[k]);
                }
                encodeTarget.put(huruf, target);
            }

            sc.nextLine();//skip line WEIGHTS:

            W = new double[inSize][outSize];//W = weights = model hasil latih
            sb.append("WEIGHTS:\n");
            for (int i = 0; i < inSize; i++) {
                baris = sc.nextLine();
                String[] kolom = baris.split(";");
                for (int j = 0; j < outSize; j++) {
                    W[i][j] = Double.parseDouble(kolom[j]);
                }
                sb.append(Arrays.toString(W[i]) + "\n");

            }

        } catch (FileNotFoundException ex) {
            ex.printStackTrace();
        }//baca MODEL Selesai

        //baca data testing dari file
        try {
            File file = fileTesting;
            Scanner sc = new Scanner(file);
            dataTesting = new ArrayList<>();
            while (sc.hasNextLine()) {
                String baris = sc.nextLine();
                String[] kolom = baris.split(":");
                char huruf = kolom[0].charAt(0);
                String[] sPola = kolom[1].split(";");
                //konversi sPola ke array integer dengan menambahkan konstan 1 di elemen pertama
                int[] pola = new int[sPola.length + 1];
                pola[0] = 1;//konstan 1 akan berpasangan dengan bias
                int k = 1;
                for (int i = 0; i < sPola.length; i++) {
                    pola[k] = Integer.parseInt(sPola[i]);
                    k++;
                }
                Data data = new Data(huruf, pola);
                dataTesting.add(data);
            }//end of while
        } catch (FileNotFoundException ex) {
            ex.printStackTrace();
        }//end of baca datatesting dari file

        //TESTING PERCEPTRON-----------------------------------_----------------
        if (W != null && dataTesting != null) {
            int t = 0;
            for (Data d : dataTesting) {
                int[] X = d.pola;
                char hurufTarget = d.huruf;
                if (X.length == W.length) {
                    sb.append("---------------------------------------------------------------------\n");
                    sb.append("Test-" + (t++) + "\n");
                    sb.append("Pola                      : " + Arrays.toString(X) + "\n");
                    sb.append("Target Testing            : " + hurufTarget + "\n");

                    //SUMMATION
                    //Hitung net
                    double[] net = new double[outSize];
                    for (int i = 0; i < inSize; i++) {
                        int Xi = X[i];
                        for (int j = 0; j < outSize; j++) {
                            double Wij = W[i][j];
                            double XiWij = Xi * Wij;
                            net[j] += XiWij;
                        }
                    }//end of hitung net

                    //ACTIVATION
                    int[] y = new int[outSize];
                    for (int j = 0; j < outSize; j++) {
                        if (net[j] > theta) {
                            y[j] = 1;
                        } else if (net[j] >= -theta) {
                            y[j] = 0;
                        } else {
                            y[j] = -1;
                        }
                    }

                    //RECOGNIZE
                    for (Character c : encodeTarget.keySet()) {
                        int[] target = encodeTarget.get(c);
                        int equals = 0;
                        for (int j = 0; j < outSize; j++) {
                            if (target[j] != y[j]) {
                                break;
                            } else {
                                equals++;
                            }
                        }
                        if (equals == outSize) {
                            sb.append("Hasil Recognize Perceptron: " + c + "\n");
                            sb.append("Kesimpulan                : ");
                            if (c.equals((Character) hurufTarget)) {
                                sb.append("TRUE\n");
                            } else {
                                sb.append("FALSE\n");
                            }
                            break;
                        }
                    }

                }//end of if (X.length == W.length) {
            }
            result = sb.toString();
        }
        return result;
    }//end of testing

}

class Data {
    char huruf;
    int[] pola;
    int[] target;

    public Data(char huruf, int[] pola) {
        this.huruf = huruf;
        this.pola = pola;
    }
    
    public String toString() {
        return "{character: " + huruf + "; pola: " + Arrays.toString(pola) + "; target: " + Arrays.toString(target) + "}\n";
    }
}
