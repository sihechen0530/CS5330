using UnityEngine;
using static UnityEngine.XR.ARSubsystems.XRCpuImage;

public class DragDropManager : MonoBehaviour
{
    public Transform handAnchor; // Hand anchor (set to the controller's position in the Inspector)
    public LayerMask wallLayer;
    public LayerMask draggableLayer; // Layer for draggable objects

    public GameObject attachObject;

    private GameObject selectedObject = null;
    private bool isDragging = false;
    private float rayLength = 10.0f;           // Length of the ray
    private float offBackground = 0.01f;
    private float draggingDistance = 0.5f;

    void Update()
    {
        // Create a ray from the hand
        Ray ray = new Ray(handAnchor.position, handAnchor.forward);

        // Check if the ray hits a draggable object
        if (Physics.Raycast(ray, out RaycastHit hit, rayLength, draggableLayer))
        {
            Debug.LogError("is hit");
            if (!isDragging && OVRInput.GetDown(OVRInput.Button.PrimaryHandTrigger)) // Grip button pressed
            {
                selectedObject = hit.collider.gameObject; // Grab the object
                isDragging = true;
            }
        }

        // Drag the object
        if (isDragging && selectedObject != null)
        {
            // Update object's position to follow the ray
            selectedObject.transform.position = ray.GetPoint(draggingDistance); // Adjust distance as needed

            // Release the object
            if (OVRInput.GetUp(OVRInput.Button.PrimaryHandTrigger)) // Grip button released
            {
                Debug.LogError("release button");

                // Optional: Add a wall-specific parent object
                // Transform wallParent = GameObject.Find("WallPhotosContainer").transform;
                // selectedPhoto.transform.SetParent(wallParent);

                // Position photo at wall hit point
                if (Physics.Raycast(ray, out RaycastHit wallHit, rayLength, wallLayer))
                {
                    //GameObject regionObject = wallHit.collider.gameObject;
                    //if (regionObject)
                    //{
                    //    selectedObject.transform.position = regionObject.transform.position;
                    //    selectedObject.transform.rotation = regionObject.transform.rotation;
                    //    selectedObject.transform.SetParent(regionObject.transform);
                    //}

                    selectedObject.transform.position = wallHit.point + wallHit.normal * offBackground;
                    selectedObject.transform.rotation = Quaternion.FromToRotation(Vector3.back, wallHit.normal);
                }

                isDragging = false;
                selectedObject = null;
            }
        }
    }
}




//using UnityEngine;
//using UnityEngine.UI;

//public class DragAndDrop : MonoBehaviour
//{
//    public Transform rightHandAnchor;         // Assign the RightHandAnchor from OVRCameraRig
//    public LayerMask wallLayer;
//    private string photoPrefabString = "PhotoPrefab";

//    private GameObject selectedPhoto = null;
//    private bool isHoldingPhoto = false;

//    private float rayLength = 10.0f;           // Length of the ray
//    private OVRInput.Button gripButton = OVRInput.Button.Two;

//    void Update()
//    {
//        // Check for ray intersection with photo
//        Ray ray = new Ray(rightHandAnchor.position, rightHandAnchor.forward);
//        //PrintRaycastHitDetails(ray);
//        if (!isHoldingPhoto && OVRInput.Get(gripButton))
//        {
//            Debug.LogError("pressed button");
//            RaycastHit[] hits = Physics.RaycastAll(ray, rayLength);

//            Debug.LogError($"Total objects hit: {hits.Length}");

//            foreach (RaycastHit hit in hits)
//            {
//                Debug.LogError("----------------------------");
//                Debug.LogError($"Object Name: {hit.collider.gameObject.name}");
//                Debug.LogError($"Object Tag: {hit.collider.gameObject.tag}");
//                Debug.LogError($"Collision Point: {hit.point}");
//                Debug.LogError($"Distance: {hit.distance}");

//                if (hit.collider.gameObject.name == photoPrefabString)
//                {
//                    selectedPhoto = hit.collider.gameObject;
//                    isHoldingPhoto = true;
//                    selectedPhoto.transform.SetParent(hit.transform);

//                }
//                // Print component types on the GameObject
//                //Component[] components = hit.collider.gameObject.GetComponents<Component>();
//                //Debug.LogError("Components on Object:");
//                //foreach (Component comp in components)
//                //{
//                //    Debug.LogError(comp.GetType().Name);
//                //}
//            }
//        }
//        // Place photo on wall when grip is released
//        if (isHoldingPhoto && OVRInput.GetUp(gripButton))
//        {
//            Debug.LogError("release button");
//            if (selectedPhoto != null)
//            {
//                // Optional: Add a wall-specific parent object
//                // Transform wallParent = GameObject.Find("WallPhotosContainer").transform;
//                // selectedPhoto.transform.SetParent(wallParent);

//                // Position photo at wall hit point
//                if (Physics.Raycast(ray,
//                             out RaycastHit wallHit,
//                             rayLength, wallLayer))
//                {
//                    selectedPhoto.transform.position = wallHit.point;

//                    // Optional: Rotate photo to face the user
//                    selectedPhoto.transform.rotation = Quaternion.LookRotation(
//                        rightHandAnchor.position - wallHit.point
//                    );
//                    isHoldingPhoto = false;
//                    selectedPhoto = null;
//                }
//            }
//        }
//    }

//    void PrintRaycastHitDetails(Ray ray)
//    {
//        RaycastHit[] hits = Physics.RaycastAll(ray, rayLength);

//        Debug.LogError($"Total objects hit: {hits.Length}");

//        foreach (RaycastHit hit in hits)
//        {
//            Debug.LogError("----------------------------");
//            Debug.LogError($"Object Name: {hit.collider.gameObject.name}");
//            Debug.LogError($"Object Tag: {hit.collider.gameObject.tag}");
//            Debug.LogError($"Collision Point: {hit.point}");
//            Debug.LogError($"Distance: {hit.distance}");

//            // Print component types on the GameObject
//            Component[] components = hit.collider.gameObject.GetComponents<Component>();
//            Debug.LogError("Components on Object:");
//            foreach (Component comp in components)
//            {
//                Debug.LogError(comp.GetType().Name);
//            }
//        }
//    }
//}