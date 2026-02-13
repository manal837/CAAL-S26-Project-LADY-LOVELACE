
#include<iostream>
#include<vector>
#include<fstream>
#include<sstream>
#include<cmath>
#include<algorithm>
#include<optional>
using namespace std;

struct Body{
public:
    double mass;
    double x, y, z;
    double vx, vy, vz;
    double ax, ay, az;

public:
    Body(double m, double dist_x, double dist_y, double dist_z,
         double velo_x, double velo_y, double velo_z){
        mass = m;
        x = dist_x;
        y = dist_y;
        z = dist_z;
        vx = velo_x;
        vy = velo_y;
        vz = velo_z;
        ax = 0.0;
        ay = 0.0;
        az = 0.0;
    }  
};

struct Node{
    double cent_x, cent_y, cent_z;
    double n;
    double mass;
    double com_x, com_y, com_z;

    Body* body;
    Node* children[8];

    Node(double cx, double cy, double cz, double size){
        cent_x = cx;
        cent_y = cy;
        cent_z = cz;
        n = size;
        mass = 0.0;
        com_x = 0.0;
        com_y = 0.0;
        com_z = 0.0;

        body = NULL;

        for (int i = 0; i < 8; i++)
            children[i] = NULL;
    }
};

int get_octant(Node* node, Body& b) {
    bool right = b.x > node->cent_x;
    bool above = b.y > node->cent_y;
    bool front = b.z > node->cent_z;
    if (!right && !above && !front) {
        return 0;
     } 
    if (!right && !above && front){
        return 1;
    }  
    if (!right && above && !front){
        return 2;
    }
    if (!right && above && front){ 
        return 3; 
    }
    if (right && !above && !front){
        return 4;
    }
    if (right && !above && front){
        return 5;
    }
    if (right && above && !front){
        return 6;
    }
    if (right && above && front){
        return 7;
    }
    return -1;
}

//this func will be called only if there are more than 1 body in a node
void subdivide(Node* node, int& node_count) {
    double new_size = node->n / 2.0;
    double half_size = new_size / 2.0;

    //cube 0: position is left,below backside
    node->children[0] = new Node(node->cent_x - half_size, node->cent_y - half_size, node->cent_z - half_size, new_size);
    node_count++;
    //cube 1: position is left, below frontside
    node->children[1] = new Node(node->cent_x - half_size, node->cent_y - half_size, node->cent_z + half_size, new_size);
    node_count++;
    //cube 2: position is left, top, backside
    node->children[2] = new Node(node->cent_x - half_size, node->cent_y + half_size, node->cent_z - half_size, new_size);
    node_count++; 
    //cube 3: position is left, top frontside
    node->children[3] = new Node(node->cent_x - half_size, node->cent_y + half_size, node->cent_z + half_size, new_size);
    node_count++;
    //cube 4: position is right, below backside
    node->children[4] = new Node(node->cent_x + half_size, node->cent_y - half_size, node->cent_z - half_size, new_size);
    node_count++; 
    //cube 5: position is right, below frontside
    node->children[5] = new Node(node->cent_x + half_size, node->cent_y - half_size, node->cent_z + half_size, new_size);
    node_count++;
    //cube 6: position is right, top backside
    node->children[6] = new Node(node->cent_x + half_size, node->cent_y + half_size, node->cent_z - half_size, new_size);
    node_count++;
    //cube 7: position is right, top frontside
    node->children[7] = new Node(node->cent_x + half_size, node->cent_y + half_size, node->cent_z + half_size, new_size);
    node_count++;
}

void insert(Node* node, Body& b, int& node_count){
    if (node->body == NULL && node->children[0] == NULL){
        node->body = &b;
        return;
    }

    if (node->children[0] == NULL){
        subdivide(node, node_count);
        if (node->body != NULL){
            int old_oct = get_octant(node, *(node->body));
            insert(node->children[old_oct], *(node->body), node_count);
            node->body = NULL;
        }
    }
    int oct = get_octant(node, b);
    insert(node->children[oct], b, node_count);
}

void compute_mass(Node* node){
    if (node == NULL) return;

    if (node->children[0] == NULL){
        if (node->body != NULL){
            node->mass = node->body->mass;
            node->com_x = node->body->x;
            node->com_y = node->body->y;
            node->com_z = node->body->z;
        }
        return;
    }

    node->mass = 0.0;
    node->com_x = node->com_y = node->com_z = 0.0;

    for (int i = 0; i < 8; i++){
        compute_mass(node->children[i]);
        node->mass += node->children[i]->mass;
        node->com_x += node->children[i]->com_x * node->children[i]->mass;
        node->com_y += node->children[i]->com_y * node->children[i]->mass;
        node->com_z += node->children[i]->com_z * node->children[i]->mass;
    }

    if (node->mass > 0){
        node->com_x /= node->mass;
        node->com_y /= node->mass;
        node->com_z /= node->mass;
    }
}

void delete_tree(Node* node){
    if (node == NULL){
    return;
    }
    for (int i = 0; i < 8; i++){
    delete_tree(node->children[i]);
    }
    delete node;
}

void calculate_force_bh(Node* node, Body& b, double theta){
    if (node == NULL || node->mass == 0){
        return;
    }
    if (node->body == &b && node->children[0] == NULL){
        return;
    }
    double dx = node->com_x - b.x;
    double dy = node->com_y - b.y;
    double dz = node->com_z - b.z;
    double r = sqrt(dx*dx + dy*dy + dz*dz);

    if (node->children[0] == NULL || (node->n / r) < theta){

        double G = 6.67430e-11;  //gavitational constant
        double softening = 1e9;    //softening length

        double r_soft = sqrt(r*r + softening*softening);
        double r_cubed = r_soft * r_soft * r_soft;

        double factor = G * node->mass / r_cubed;

        b.ax += factor * dx;
        b.ay += factor * dy;
        b.az += factor * dz;
    }
    else{
        for (int i = 0; i < 8; i++)
            calculate_force_bh(node->children[i], b, theta);
    }
}

//this funct computes 
void computeBoundingBox(const vector<Body>& bodies, double& cent_x, double& cent_y, double& cent_z, double& size) {
    double min_x = 1e100, min_y = 1e100, min_z = 1e100;
    double max_x = -1e100, max_y = -1e100, max_z = -1e100;
    
    for (const auto& body : bodies) {
        min_x = std::min(min_x, body.x);
        min_y = std::min(min_y, body.y);
        min_z = std::min(min_z, body.z);
        max_x = std::max(max_x, body.x);
        max_y = std::max(max_y, body.y);
        max_z = std::max(max_z, body.z);
    }
    
    //compute the center
    cent_x = (min_x + max_x) / 2.0;
    cent_y = (min_y + max_y) / 2.0;
    cent_z = (min_z + max_z) / 2.0;
    
    double width_x = max_x - min_x;
    double width_y = max_y - min_y;
    double width_z = max_z - min_z;
    double max_width = max(std::max(width_x, width_y), width_z);
    double padding = max_width * 0.01;  //added 1% padding to make sure all bodies fit 
    size = max_width + 2 * padding;
    
    //making sure size is positive
    if (size <= 0) size = 1e10; 
}


void calculate_forces_bh(vector<Body>& bodies,bool print_tree_info = false ) {
    int n = bodies.size();
    for (int i = 0; i < n; i++) {
        bodies[i].ax = 0.0;
        bodies[i].ay = 0.0;
        bodies[i].az = 0.0;
    }

    double cent_x, cent_y, cent_z, size;
    computeBoundingBox(bodies, cent_x, cent_y, cent_z, size);
    
    int node_count = 0;
    Node* root = new Node(cent_x, cent_y, cent_z, size);
    node_count++;
    
    for (int i = 0; i < n; i++){
    
        insert(root, bodies[i], node_count);
    compute_mass(root);
    }

    if(print_tree_info){
    cout<<"no. of tree nodes created:"<< node_count<<endl;
   }

    double theta = 0.5;
    for (int i = 0; i < n; i++)
        calculate_force_bh(root, bodies[i], theta);
    
    delete_tree(root);
}

void kick_half_step(vector<Body>& bodies, double dt){
    for (int i = 0; i < bodies.size(); i++){
        bodies[i].vx += 0.5 * bodies[i].ax * dt;
        bodies[i].vy += 0.5 * bodies[i].ay * dt;
        bodies[i].vz += 0.5 * bodies[i].az * dt;
    }
}

void drift(vector<Body>& bodies, double dt)
{
    for (int i = 0; i < bodies.size(); i++){
        bodies[i].x += bodies[i].vx * dt;
        bodies[i].y += bodies[i].vy * dt;
        bodies[i].z += bodies[i].vz * dt;
    }
}

// funct to calculate total energy   -- for debugging and test to see if energy is conserved 
double calculate_energy(vector<Body>& bodies){
    double kinetic = 0.0;
    double potential = 0.0;
    const double G = 6.67430e-11;
    int n = bodies.size();
    // calculate ke 
    for (int i = 0; i < n; i++){
        double v2 = bodies[i].vx * bodies[i].vx + 
                    bodies[i].vy * bodies[i].vy + 
                    bodies[i].vz * bodies[i].vz;
        kinetic += 0.5 * bodies[i].mass * v2;
    }
    
    //calculate pe
    for (int i = 0; i < n; i++){
        for (int j = i+1; j < n; j++){
            double dx = bodies[i].x - bodies[j].x;
            double dy = bodies[i].y - bodies[j].y;
            double dz = bodies[i].z - bodies[j].z;
            double r = sqrt(dx*dx + dy*dy + dz*dz);
            potential -= G * bodies[i].mass * bodies[j].mass / r;
        }
    }
    
    return kinetic + potential;
}

//func added to calculate and show com energy and momentum
void test(const vector<Body>& bodies, int step) {
    double total_mass = 0.0;
    double com_x = 0.0, com_y = 0.0, com_z = 0.0;
    double total_momentum_x = 0.0, total_momentum_y = 0.0, total_momentum_z = 0.0;

    for (const auto& body : bodies) {
        total_mass += body.mass;
        com_x += body.mass * body.x;
        com_y += body.mass * body.y;
        com_z += body.mass * body.z;
        total_momentum_x += body.mass * body.vx;
        total_momentum_y += body.mass * body.vy;
        total_momentum_z += body.mass * body.vz;
    }

    com_x /= total_mass;
    com_y /= total_mass;
    com_z /= total_mass;

    double energy = calculate_energy(bodies);

    cout<<"Step: " <<step<< "Energy: " <<energy<< "COM:" <<com_x<< ", " <<com_y<< ", " <<com_z<< "Momentum:" <<total_momentum_x<< ", " <<total_momentum_y<< ", " <<total_momentum_z<<endl;
    cout<<"Bodies state";
    for (int i = 0; i < bodies.size(); i++) {
        cout<< "Body "<< i << ":"<< "x =" <<bodies[i].x << " y =" << bodies[i].y << " z =" << bodies[i].z << " | vx =" << bodies[i].vx << " vy =" << bodies[i].vy << " vz =" << bodies[i].vz << endl;
    }

}

// func for reading and using values from csv
void opencsv(const string& filename, vector<Body>& bodies){
    ifstream file(filename);

    if (!file.is_open()){
        cout<<"error, file cannot be opened"<<endl;
        return;
    }

    string line;
    getline(file, line); //skipping the first line since it has headings

    while (getline(file, line)){
        stringstream ss(line);
        string value;
        vector<double> data;

        while(getline(ss, value, ',')){
            data.push_back(stod(value));
        }

        if (data.size() == 7){
            Body b(data[0],data[1],data[2],data[3],data[4],data[5],data[6]);
            bodies.push_back(b);
        }
    }
    file.close();
}

int main(){
    vector<Body> bodies;
    double dt = 8640;
    int timesteps = 100; 

    opencsv("stable_random_system100.csv", bodies);
    cout << "no. of bodies from CSV " << bodies.size() << endl;
    cout << "initial state of all bodies" << endl;
    test(bodies, 0);

    calculate_forces_bh(bodies, true);

    double initial_energy = calculate_energy(bodies);
    cout << "initial total energy " << initial_energy << endl;

    for (int step = 1; step <= timesteps; step++){
        kick_half_step(bodies, dt);
        drift(bodies, dt);
        calculate_forces_bh(bodies);
        kick_half_step(bodies, dt);

        if (step % 10 == 0){
            test(bodies, step);
        }
    }
    cout << "final state after" << timesteps << "steps" << endl;
    test(bodies, timesteps);

    double final_energy = calculate_energy(bodies);
    cout << "Final total energy" << final_energy << endl;
    cout << "energy drift " << (final_energy - initial_energy) / initial_energy * 100 << "%" << endl;
