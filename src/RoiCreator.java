import ij.gui.PolygonRoi;
import ij.gui.Roi;

import java.awt.Polygon;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Converts deserialized predictions into ImageJ polygon ROIs. */
public final class RoiCreator {
    private RoiCreator() {
    }

    public static List<PolygonRoi> createRois(List<Prediction> predictions) {
        if (predictions == null) {
            return Collections.emptyList();
        }

        List<PolygonRoi> rois = new ArrayList<PolygonRoi>();

        for (int i = 0; i < predictions.size(); i++) {
            Prediction prediction = predictions.get(i);
            Polygon polygon = toPolygon(prediction);

            if (polygon == null) {
                continue;
            }

            PolygonRoi roi = new PolygonRoi(polygon, Roi.POLYGON);
            roi.setName("AI Prediction " + (i + 1));
            rois.add(roi);
        }

        return rois;
    }

    private static Polygon toPolygon(Prediction prediction) {
        if (prediction == null || prediction.polygon == null) {
            return null;
        }

        Polygon polygon = new Polygon();

        for (List<Integer> point : prediction.polygon) {
            if (point == null || point.size() != 2
                    || point.get(0) == null || point.get(1) == null) {
                continue;
            }

            polygon.addPoint(point.get(0), point.get(1));
        }

        return polygon.npoints >= 3 ? polygon : null;
    }
}
