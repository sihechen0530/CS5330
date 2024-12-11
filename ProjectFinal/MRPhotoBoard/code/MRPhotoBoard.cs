using System.IO;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Android;

public class MRPhotoBoard : MonoBehaviour
{
    public OVRHand hand;
    private enum InteractionState { Idle, Drawing, Menu, Dragging }
    private InteractionState currentInteractionState = InteractionState.Idle;
    private enum ControlState { None, Start, Hold, Release }
    private ControlState currentControlState = ControlState.None;

    private float pinchStartThreshold = 0.8f;
    private float pinchReleaseThreshold = 0.2f;

    public LayerMask wallLayer;               // Layer for the wall
    public LayerMask photoLayer;               // Layer for the wall
    public GameObject hitMarkerPrefab;        // Prefab for the hit marker
    public float rayLength = 10.0f;           // Length of the ray
    public Color rayColor = Color.green;      // Color for the ray visualization
    public GameObject currentRectangle; // Prefab for the rectangle overlay

    private Vector3 startPoint;
    private Vector3 endPoint;
    private Ray ray;

    private GameObject selectedPhoto = null;
    private float offBackground = 0.01f;
    private float draggingDistance = 0.5f;


    private float waveThreshold = 0.2f;  // Minimum distance for a wave
    private float timeWindow = 0.15f;     // Maximum time to complete the wave

    private Vector3 previousPosition;
    private float startTime;


    public GameObject photoPad; // The photo pad UI
    public GameObject photoPrefab; // Prefab for displaying a photo
    private string photoDirectory = "/storage/emulated/0/Download"; // Directory for photos
    private float offsetDistance = 0.3f; // Distance from hand
    private Vector3 rotationOffset = new Vector3(0, 0, 0); // Rotation to face the user

    void Start()
    {
        photoPad.SetActive(false);
        if (!Permission.HasUserAuthorizedPermission(Permission.ExternalStorageRead))
        {
            Permission.RequestUserPermission(Permission.ExternalStorageRead);
        }

        if (!Permission.HasUserAuthorizedPermission(Permission.ExternalStorageWrite))
        {
            Permission.RequestUserPermission(Permission.ExternalStorageWrite);
        }
    }
    private void Update()
    {
        Transform handAnchor = hand.PointerPose;
        ray = new Ray(handAnchor.position, handAnchor.forward);

        UpdateControlStateByPinch();

        DrawDebugRay(ray);

        // TODO: place hit marker at raycast
        if (currentControlState == ControlState.Start)
        {
            if (currentInteractionState == InteractionState.Idle)
            {
                currentInteractionState = InteractionState.Drawing;
                if (Physics.Raycast(ray, out RaycastHit wallHit, rayLength, wallLayer))
                {
                    startPoint = wallHit.point;
                }
            }
            else if (currentInteractionState == InteractionState.Menu)
            {
                if (TrySelectPhoto(ray, photoLayer))
                {
                    currentInteractionState = InteractionState.Dragging;
                }
            }
        }
        else if (currentControlState == ControlState.Hold)
        {
            if (currentInteractionState == InteractionState.Drawing)
            {
                if (Physics.Raycast(ray, out RaycastHit wallHit, rayLength, wallLayer))
                {
                    endPoint = wallHit.point;
                }
                UpdateRectangle(currentRectangle, startPoint, endPoint, wallHit.normal);
            }
            else if (currentInteractionState == InteractionState.Dragging && selectedPhoto != null)
            {
                selectedPhoto.transform.position = ray.GetPoint(draggingDistance); // Adjust distance as needed
            }
        }
        else if (currentControlState == ControlState.Release)
        {
            if (currentInteractionState == InteractionState.Drawing)
            {
                currentInteractionState = InteractionState.Idle;
            }
            else if (currentInteractionState == InteractionState.Dragging)
            {
                if (Physics.Raycast(ray, out RaycastHit wallHit, rayLength, wallLayer))
                {
                    selectedPhoto.transform.position = wallHit.point + wallHit.normal * offBackground;
                    selectedPhoto.transform.rotation = Quaternion.FromToRotation(Vector3.back, wallHit.normal);
                }
                selectedPhoto = null;
                currentInteractionState = InteractionState.Menu;
            }
        }

        string swipeDirection = GetSwipeDirection();
        if (swipeDirection == "LEFT")
        {
            LoadPhotos();
            PositionPhotoPad();
            photoPad.SetActive(true);
            currentInteractionState = InteractionState.Menu;
        }
        if (swipeDirection == "RIGHT")
        {
            photoPad.SetActive(false);
            currentInteractionState = InteractionState.Idle;
        }
    }

    private void UpdateControlStateByPinch()
    {
        float pinchStrength = hand.GetFingerPinchStrength(OVRHand.HandFinger.Index);
        switch (currentControlState)
        {
            case ControlState.None:
                if (pinchStrength > pinchStartThreshold)
                {
                    currentControlState = ControlState.Start;
                    Debug.Log("Pinch Start");
                }
                break;

            case ControlState.Start:
                if (pinchStrength > pinchStartThreshold)
                {
                    currentControlState = ControlState.Hold;
                    Debug.Log("Pinch Hold");
                }
                break;

            case ControlState.Hold:
                if (pinchStrength < pinchReleaseThreshold)
                {
                    currentControlState = ControlState.Release;
                    Debug.Log("Pinch Release");
                }
                break;

            case ControlState.Release:
                if (pinchStrength < pinchReleaseThreshold)
                {
                    currentControlState = ControlState.None;
                    Debug.Log("Pinch Ended");
                }
                break;
        }
    }

    private string GetSwipeDirection()
    {
        string direction = null;
        if (hand.IsTracked)
        {
            Vector3 currentPosition = hand.transform.position;

            if (Time.time - startTime > timeWindow)
            {
                // Reset if the time window is exceeded
                startTime = Time.time;
                previousPosition = currentPosition;
            }

            if (Vector3.Distance(previousPosition, currentPosition) > waveThreshold)
            {
                Vector3 swipeDirection = currentPosition - previousPosition;

                // Horizontal Swipe
                if (swipeDirection.x < 0)
                {
                    Debug.Log("Swipe Left");
                    direction = "LEFT";
                }
                else
                {
                    Debug.Log("Swipe Right");
                    photoPad.SetActive(false);
                    direction = "RIGHT";
                }
                startTime = Time.time; // Reset time for the next swipe
            }
        }
        return direction;
    }

    private bool TrySelectPhoto(Ray ray, LayerMask photoLayer)
    {
        if (Physics.Raycast(ray, out RaycastHit photoHit, rayLength, photoLayer))
        {
            //DrawDebugRay(ray.origin, photoHit.point );
            selectedPhoto = photoHit.collider.gameObject;
            return true;
        }
        selectedPhoto = null;
        return false;
    }

    private void UpdateRectangle(GameObject rectangle, Vector3 point1, Vector3 point2, Vector3 normal)
    {
        Vector3 midpoint = (point1 + point2) / 2;

        // Calculate scale (width and height)
        float width = Vector3.Distance(new Vector3(point1.x, 0, point1.z), new Vector3(point2.x, 0, point2.z));
        float height = Mathf.Abs(point2.y - point1.y);
        Vector3 scale = new Vector3(width, height, 1f);

        Quaternion rotation = Quaternion.FromToRotation(Vector3.back, normal);

        // Apply transformations
        rectangle.transform.position = midpoint;
        rectangle.transform.rotation = rotation;
        rectangle.transform.localScale = scale;

    }

    private void LoadPhotos()
    {
        // Clear existing photos
        Transform gridParent = photoPad.GetComponent<GridLayoutGroup>().transform;
        foreach (Transform child in gridParent)
        {
            Destroy(child.gameObject);
        }

        // Load all photos from the specified directory
        string path = photoDirectory;
        if (!Directory.Exists(path))
        {
            Debug.LogError("Photo directory not found: " + path);
            return;
        }

        string[] photoFiles = Directory.GetFiles(path, "*.jpeg"); // or "*.jpg"
        foreach (string photoFile in photoFiles)
        {
            Debug.LogError("loading photo" + photoFile);
            CreatePhotoItem(photoFile);
        }
    }

    private void CreatePhotoItem(string filePath)
    {
        try
        {
            // Load the image
            byte[] fileData = File.ReadAllBytes(filePath);
            Texture2D texture = new Texture2D(2, 2, TextureFormat.RGBA32, false);
            if (texture.LoadImage(fileData))
            {
                // Create a photo item
                Transform gridParent = photoPad.GetComponent<GridLayoutGroup>().transform;
                GameObject photoItem = Instantiate(photoPrefab, gridParent);

                // Get Image component
                Image imgComponent = photoItem.GetComponent<Image>();
                if (imgComponent != null)
                {
                    imgComponent.color = Color.white;
                    // Create sprite with original texture dimensions
                    Sprite photoSprite = Sprite.Create(
                        texture,
                        new Rect(0, 0, texture.width, texture.height),
                        new Vector2(0.5f, 0.5f), // Center pivot
                        100f // Pixels per unit (adjust as needed)
                    );

                    imgComponent.sprite = photoSprite;

                    // Optional: Maintain aspect ratio
                    imgComponent.preserveAspect = true;
                }
            }
            else
            {
                Debug.LogError("Failed to load image: " + filePath);
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError("Error creating photo item: " + e.Message);
        }
    }

    private void PositionPhotoPad()
    {
        if (hand != null)
        {
            // Get the position of the right hand
            Vector3 handPosition = hand.PointerPose.position;

            // Calculate the position slightly to the side of the hand
            Vector3 padPosition = handPosition +
                hand.PointerPose.forward * offsetDistance;

            // Set the position of the photo pad
            photoPad.transform.position = padPosition;

            // Rotate the pad to face the user
            photoPad.transform.rotation = Quaternion.Euler(
                hand.PointerPose.rotation.eulerAngles + rotationOffset
            );
        }
        else
        {
            Debug.LogWarning("Right hand reference is not set!");
        }
    }

    private void DrawDebugRay(Ray ray)
    {
        if (Physics.Raycast(ray, out RaycastHit hitInfo, rayLength))
        {
            hitMarkerPrefab.transform.position = hitInfo.transform.position;
            Vector3 backwardDirection = -ray.direction.normalized;

            // Adjust the hit marker position by moving it slightly backward
            float offsetDistance = 0.05f; // Adjust this value as needed
            hitMarkerPrefab.transform.position = hitInfo.point + backwardDirection * offsetDistance;
        }
    }

}


