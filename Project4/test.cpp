#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <vector>

// Function to check OpenGL shader compilation errors
void checkCompileErrors(GLuint shader, std::string type) {
  GLint success;
  GLchar infoLog[1024];
  if (type != "PROGRAM") {
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
      glGetShaderInfoLog(shader, 1024, NULL, infoLog);
      std::cerr << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n"
                << infoLog << std::endl;
    }
  } else {
    glGetProgramiv(shader, GL_LINK_STATUS, &success);
    if (!success) {
      glGetProgramInfoLog(shader, 1024, NULL, infoLog);
      std::cerr << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n"
                << infoLog << std::endl;
    }
  }
}

// Function to load the .obj model using Assimp
void loadModel(const std::string &path, std::vector<float> &vertices,
               std::vector<float> &normals) {
  Assimp::Importer importer;
  const aiScene *scene =
      importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs);

  if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE ||
      !scene->mRootNode) {
    std::cerr << "ERROR::ASSIMP:: " << importer.GetErrorString() << std::endl;
    return;
  }

  aiMesh *mesh = scene->mMeshes[0]; // Assuming the model has only one mesh
  for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
    aiVector3D vertex = mesh->mVertices[i];
    aiVector3D normal = mesh->mNormals[i];

    // Push vertex and normal data into the vectors
    vertices.push_back(vertex.x);
    vertices.push_back(vertex.y);
    vertices.push_back(vertex.z);

    normals.push_back(normal.x);
    normals.push_back(normal.y);
    normals.push_back(normal.z);
  }
}

// Vertex shader
const char *vertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec3 position;
    layout (location = 1) in vec3 normal;

    out vec3 FragPos;
    out vec3 Normal;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    void main() {
        FragPos = vec3(model * vec4(position, 1.0));
        Normal = mat3(transpose(inverse(model))) * normal;
        gl_Position = projection * view * vec4(FragPos, 1.0);
    }
)";

// Fragment shader
const char *fragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;

    in vec3 FragPos;
    in vec3 Normal;

    uniform vec3 lightPos;
    uniform vec3 viewPos;
    uniform vec3 lightColor;
    uniform vec3 objectColor;

    void main() {
        // Ambient lighting
        float ambientStrength = 0.1;
        vec3 ambient = ambientStrength * lightColor;

        // Diffuse lighting
        vec3 norm = normalize(Normal);
        vec3 lightDir = normalize(lightPos - FragPos);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = diff * lightColor;

        // Specular lighting
        float specularStrength = 0.5;
        vec3 viewDir = normalize(viewPos - FragPos);
        vec3 reflectDir = reflect(-lightDir, norm);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
        vec3 specular = specularStrength * spec * lightColor;

        vec3 result = (ambient + diffuse + specular) * objectColor;
        FragColor = vec4(result, 1.0);
    }
)";

int main() {
  // Initialize GLFW
  if (!glfwInit()) {
    std::cerr << "Failed to initialize GLFW" << std::endl;
    return -1;
  }
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glEnable(GL_DEPTH_TEST);

  // Create a window
  GLFWwindow *window =
      glfwCreateWindow(800, 600, "Shaded Cube with Assimp", NULL, NULL);
  if (!window) {
    std::cerr << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);

  // Initialize GLEW
  glewExperimental = GL_TRUE;
  if (glewInit() != GLEW_OK) {
    std::cerr << "Failed to initialize GLEW" << std::endl;
    return -1;
  }

  // Compile and link shaders
  GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
  glCompileShader(vertexShader);
  checkCompileErrors(vertexShader, "VERTEX");

  GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
  glCompileShader(fragmentShader);
  checkCompileErrors(fragmentShader, "FRAGMENT");

  GLuint shaderProgram = glCreateProgram();
  glAttachShader(shaderProgram, vertexShader);
  glAttachShader(shaderProgram, fragmentShader);
  glLinkProgram(shaderProgram);
  checkCompileErrors(shaderProgram, "PROGRAM");

  glDeleteShader(vertexShader);
  glDeleteShader(fragmentShader);

  // Load model from .obj file
  std::vector<float> vertices;
  std::vector<float> normals;
  loadModel("../resources/triangle.obj", vertices, normals);

  GLuint VAO, VBO;
  glGenVertexArrays(1, &VAO);
  glGenBuffers(1, &VBO);

  glBindVertexArray(VAO);
  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER,
               vertices.size() * sizeof(float) + normals.size() * sizeof(float),
               NULL, GL_STATIC_DRAW);
  glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.size() * sizeof(float),
                  vertices.data());
  glBufferSubData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float),
                  normals.size() * sizeof(float), normals.data());

  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);

  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                        (void *)(3 * sizeof(float)));
  glEnableVertexAttribArray(1);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  // Lighting and camera setup
  glm::vec3 lightPos(1.2f, 1.0f, 2.0f);
  glm::vec3 viewPos(0.0f, 0.0f, 3.0f);

  while (!glfwWindowShouldClose(window)) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(shaderProgram);
    glUniform3fv(glGetUniformLocation(shaderProgram, "lightPos"), 1,
                 glm::value_ptr(lightPos));
    glUniform3fv(glGetUniformLocation(shaderProgram, "viewPos"), 1,
                 glm::value_ptr(viewPos));
    glUniform3f(glGetUniformLocation(shaderProgram, "lightColor"), 1.0f, 1.0f,
                1.0f);
    glUniform3f(glGetUniformLocation(shaderProgram, "objectColor"), 1.0f, 0.5f,
                0.31f);

    glm::mat4 model = glm::mat4(1.0f);
    glm::mat4 view = glm::lookAt(viewPos, glm::vec3(0.0f, 0.0f, 0.0f),
                                 glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 projection =
        glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);

    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1,
                       GL_FALSE, glm::value_ptr(model));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE,
                       glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1,
                       GL_FALSE, glm::value_ptr(projection));

    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, vertices.size() / 3);

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  glDeleteVertexArrays(1, &VAO);
  glDeleteBuffers(1, &VBO);

  glfwTerminate();
  return 0;
}
