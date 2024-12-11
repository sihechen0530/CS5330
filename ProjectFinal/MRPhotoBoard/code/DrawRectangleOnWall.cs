using System.IO;
using UnityEngine;

public class HandTrackingRaycaster : MonoBehaviour
{
    public Transform rightHandAnchor;         // Assign the RightHandAnchor from OVRCameraRig
    public OVRHand hand; // Assign LeftHandAnchor or RightHandAnchor
    private enum ControlState { None, Start, Hold, Release }
    private ControlState currentState = ControlState.None;
    private float pinchStartThreshold = 0.8f;
    private float pinchReleaseThreshold = 0.2f;

    public LayerMask wallLayer;               // Layer for the wall
    public GameObject hitMarkerPrefab;        // Prefab for the hit marker
    public float rayLength = 10.0f;           // Length of the ray
    public Color rayColor = Color.green;      // Color for the ray visualization
    public GameObject currentRectangle; // Prefab for the rectangle overlay

    private Vector3 startPoint;
    private Vector3 endPoint;
    private Ray ray;
    private RaycastHit hitInfo;
    private bool isHit = false;

    private void Update()
    {
        if (hand != null)
        {
            rightHandAnchor = hand.PointerPose;
        }
        // step 1: update ray
        ray = new Ray(rightHandAnchor.position, rightHandAnchor.forward);

        // step 2: Cast a ray from the right hand (you can add a condition to use left or right hand based on input)
        CastRayFromHand(rightHandAnchor);

        UpdateControlState();

        //UpdateControlStateByPinch();

        // step 3: get starting point
        if (currentState == ControlState.Start)
        {
            startPoint = hitInfo.point;
        }
        else if (currentState == ControlState.Hold)
        {
            // Update rectangle end point
            if (isHit)
            {
                endPoint = hitInfo.point;
                UpdateRectangle(currentRectangle, startPoint, endPoint, hitInfo.normal);
            }
        }
    }

    void UpdateControlStateByPinch()
    {
        float pinchStrength = hand.GetFingerPinchStrength(OVRHand.HandFinger.Index);

        switch (currentState)
        {
            case ControlState.None:
                if (pinchStrength > pinchStartThreshold)
                {
                    currentState = ControlState.Start;
                    Debug.Log("Pinch Start");
                }
                break;

            case ControlState.Start:
                if (pinchStrength > pinchStartThreshold)
                {
                    currentState = ControlState.Hold;
                    Debug.Log("Pinch Hold");
                }
                break;

            case ControlState.Hold:
                if (pinchStrength < pinchReleaseThreshold)
                {
                    currentState = ControlState.Release;
                    Debug.Log("Pinch Release");
                }
                break;

            case ControlState.Release:
                if (pinchStrength < pinchReleaseThreshold)
                {
                    currentState = ControlState.None;
                    Debug.Log("Pinch Ended");
                }
                break;
        }
    }

    void UpdateControlState()
    {
        switch (currentState)
        {
            case ControlState.None:
                if (OVRInput.GetDown(OVRInput.Button.PrimaryIndexTrigger))
                {
                    currentState = ControlState.Start;
                    Debug.Log("Pinch Start");
                }
                break;

            case ControlState.Start:
                if (OVRInput.Get(OVRInput.Button.PrimaryIndexTrigger))
                {
                    currentState = ControlState.Hold;
                    Debug.Log("Pinch Hold");
                }
                break;

            case ControlState.Hold:
                if (OVRInput.GetUp(OVRInput.Button.PrimaryIndexTrigger))
                {
                    currentState = ControlState.Release;
                    Debug.Log("Pinch Release");
                }
                break;

            case ControlState.Release:
                if (OVRInput.GetUp(OVRInput.Button.PrimaryIndexTrigger))
                {
                    currentState = ControlState.None;
                    Debug.Log("Pinch Ended");
                }
                break;
        }
    }

    void UpdateRectangle(GameObject rectangle, Vector3 point1, Vector3 point2, Vector3 normal)
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

        //UpdateTexture(rectangle);
    }

    private void CastRayFromHand(Transform handAnchor)
    {
        // TODO: remove line renderer; only use hit marker
        // Check if the hand anchor is valid
        if (handAnchor == null)
        {
            isHit = false;
            return;
        }

        // Check for collision with the wall layer
        if (Physics.Raycast(ray, out hitInfo, rayLength, wallLayer))
        {

            // Place the hit marker at the impact point
            PlaceHitMarker(hitInfo.point);

            isHit = true;
            return;
        }
        isHit = false;
    }

    private void PlaceHitMarker(Vector3 position)
    {

        hitMarkerPrefab.transform.position = position;
        Vector3 backwardDirection = -ray.direction.normalized;

        // Adjust the hit marker position by moving it slightly backward
        float offsetDistance = 0.05f; // Adjust this value as needed
        hitMarkerPrefab.transform.position = position + backwardDirection * offsetDistance;
    }

    private void UpdateTexture(GameObject rectangle)
    {

        string filePath = "/storage/emulated/0/Download/premium_photo-1673967831980-1d377baaded2.jpeg";
        byte[] fileData = File.ReadAllBytes(filePath);
        Texture2D imageToApply = new Texture2D(2, 2, TextureFormat.RGBA32, false);
        imageToApply.LoadImage(fileData);
        Rect region = new Rect(0, 0, 100, 100); // The region on the texture to modify (x, y, width, height)

        Renderer renderer = rectangle.GetComponent<Renderer>();
        // Assign the modified texture back to the material
        renderer.material.mainTexture = imageToApply;

    }
}


