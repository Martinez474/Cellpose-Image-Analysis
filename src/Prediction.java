import java.util.List;

/** One prediction produced by the segmentation model. */
public class Prediction {
    public String label;
    public double confidence;
    public List<List<Integer>> polygon;
}
