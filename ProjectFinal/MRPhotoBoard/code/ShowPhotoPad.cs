using UnityEngine;
using UnityEngine.Android;
using UnityEngine.UI;
using System.IO;

public class PhotoPadController : MonoBehaviour
{
    public GameObject photoPad; // The photo pad UI
    public GameObject photoPrefab; // Prefab for displaying a photo
    public Transform rightHandAnchor;         // Assign the RightHandAnchor from OVRCameraRig
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
        if (OVRInput.GetDown(OVRInput.Button.One))
        {
            ShowPhotoPad();
        }
    }

    public void ShowPhotoPad()
    {
        // Toggle visibility
        photoPad.SetActive(!photoPad.activeSelf);
        if (photoPad.activeSelf)
        {
            LoadPhotos();
            PositionPhotoPad();
        }
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
                else
                {
                    Debug.LogError("Photo prefab is missing Image component!");
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
        if (rightHandAnchor != null)
        {
            // Get the position of the right hand
            Vector3 handPosition = rightHandAnchor.transform.position;

            // Calculate the position slightly to the side of the hand
            Vector3 padPosition = handPosition +
                rightHandAnchor.transform.forward * offsetDistance;

            // Set the position of the photo pad
            photoPad.transform.position = padPosition;

            // Rotate the pad to face the user
            photoPad.transform.rotation = Quaternion.Euler(
                rightHandAnchor.transform.rotation.eulerAngles + rotationOffset
            );
        }
        else
        {
            Debug.LogWarning("Right hand reference is not set!");
        }
    }
}
