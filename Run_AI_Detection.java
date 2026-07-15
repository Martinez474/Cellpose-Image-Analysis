import ij.IJ;
import ij.ImagePlus;
import ij.gui.PolygonRoi;
import ij.gui.Roi;
import ij.io.OpenDialog;
import ij.plugin.PlugIn;
import ij.plugin.frame.RoiManager;

import java.awt.Polygon;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;

public class Run_AI_Detection implements PlugIn {

    @Override
    public void run(String arg) {
        ImagePlus image = IJ.getImage();

        if (image == null) {
            IJ.showMessage("Run AI Detection", "No image is currently open.");
            return;
        }

        OpenDialog dialog = new OpenDialog("Choose predictions.json", null);
        String directory = dialog.getDirectory();
        String filename = dialog.getFileName();

        if (filename == null) {
            return;
        }

        String path = directory + filename;

        try {
            String json = readFile(path);
            ArrayList<Polygon> polygons = parsePolygons(json);

            if (polygons.size() == 0) {
                IJ.showMessage("Run AI Detection", "No polygons found in JSON.");
                return;
            }

            RoiManager roiManager = RoiManager.getInstance();
            if (roiManager == null) {
                roiManager = new RoiManager();
            }

            for (int i = 0; i < polygons.size(); i++) {
                PolygonRoi roi = new PolygonRoi(polygons.get(i), Roi.POLYGON);
                roi.setName("AI Prediction " + (i + 1));
                roiManager.addRoi(roi);
            }

            IJ.showMessage(
                    "Run AI Detection",
                    "Added " + polygons.size() + " AI prediction ROIs."
            );

        } catch (Exception e) {
            IJ.showMessage("Run AI Detection Error", e.getMessage());
            IJ.log(e.toString());
        }
    }

    private String readFile(String path) throws Exception {
        BufferedReader reader = new BufferedReader(new FileReader(path));
        StringBuilder builder = new StringBuilder();

        String line;
        while ((line = reader.readLine()) != null) {
            builder.append(line);
        }

        reader.close();
        return builder.toString();
    }

    private ArrayList<Polygon> parsePolygons(String json) {
        ArrayList<Polygon> polygons = new ArrayList<Polygon>();

        int searchIndex = 0;

        while (true) {
            int polygonKey = json.indexOf("\"polygon\"", searchIndex);

            if (polygonKey == -1) {
                break;
            }

            int arrayStart = json.indexOf("[[", polygonKey);
            int arrayEnd = json.indexOf("]]", arrayStart);

            if (arrayStart == -1 || arrayEnd == -1) {
                break;
            }

            String polygonText = json.substring(arrayStart + 2, arrayEnd);
            String[] pointTexts = polygonText.split("\\],\\s*\\[");

            Polygon polygon = new Polygon();

            for (int i = 0; i < pointTexts.length; i++) {
                String pointText = pointTexts[i].replace("[", "").replace("]", "");
                String[] xy = pointText.split(",");

                if (xy.length == 2) {
                    int x = Integer.parseInt(xy[0].trim());
                    int y = Integer.parseInt(xy[1].trim());
                    polygon.addPoint(x, y);
                }
            }

            if (polygon.npoints > 0) {
                polygons.add(polygon);
            }

            searchIndex = arrayEnd + 2;
        }

        return polygons;
    }
}