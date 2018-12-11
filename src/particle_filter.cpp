/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

	// Set the number of particles. Initialize all particles to first position (based on estimates of
	// x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 100;
	default_random_engine gen;
	double std_x, std_y, std_theta; // Standard deviations for x, y, and theta

	//Set standard deviations for x, y, and theta
	 std_x = std[0];
	 std_y = std[1];
	 std_theta = std[2];

	//Create normal distributions for x, y and theta
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);


	for (long i = 0; i < num_particles; ++i)
	{
		//here "gen" is the random engine initialized earlier.
		particles.push_back(Particle());
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
		particles[i].weight = 1;
	}
	is_initialized = true;
	flag = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	// http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	// http://www.cplusplus.com/reference/random/default_random_engine/

	std::default_random_engine engine;
	default_random_engine gen;
	double std_x, std_y, std_yaw; // Standard deviations for x, y, and yaw

	//Set standard deviations for x, y, and yaw
	 std_x   = std_pos[0];
	 std_y   = std_pos[1];
	 std_yaw = std_pos[2];

	 double new_x;
	 double new_y;
	 double new_theta;
	 for(long i = 0; i < num_particles; ++i)
	 {
		 if(fabs(yaw_rate) >= .0001)
		 {
			 double old_theta = particles[i].theta;
			 double velocity_to_yaw_rate = velocity / yaw_rate;
			 new_theta = particles[i].theta + delta_t * yaw_rate;
			 new_x = particles[i].x + velocity_to_yaw_rate * (sin(new_theta) - sin(old_theta));
			 new_y = particles[i].y + velocity_to_yaw_rate * (cos(old_theta) - cos(new_theta));
		 }
		 else
		 {
			 new_x = particles[i].x + velocity * delta_t * cos(particles[i].theta);
			 new_y = particles[i].y + velocity * delta_t * sin(particles[i].theta);
			 new_theta = particles[i].theta;
		 }

		 //Create normal distributions for x, y and theta
		 normal_distribution<double> dist_x(0, std_x);
		 normal_distribution<double> dist_y(0, std_y);
		 normal_distribution<double> dist_theta(0, std_yaw);

		 //Update particle state.
		 particles[i].x = new_x + dist_x(gen);
		 particles[i].y = new_y + dist_y(gen);
		 particles[i].theta = new_theta + dist_theta(gen);

	 }

}



void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the
	// observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	// implement this method and use it as a helper during the updateWeights phase.

	for(size_t i = 0; i < observations.size(); ++i)
	{
		double distance = dist(observations[i].x, observations[i].y, predicted[0].x, predicted[0].y);
		observations[i].id = 0;
		for(size_t j = 1; j < predicted.size(); ++j)
		{
			double new_dist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
			if(new_dist < distance)
			{
				distance = new_dist;
				observations[i].id = j;
			}
		}
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	// more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	// according to the MAP'S coordinate system. You will need to transform between the two systems.
	// Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	// The following is a good resource for the theory:
	// https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	// and the following is a good resource for the actual equation to implement (look at equation
	// 3.33
	// http://planning.cs.uiuc.edu/node99.html

	max_weight = -1;
	weights.clear();
	for(auto &particle : particles)
	{
		std::vector<LandmarkObs> predicted_distances;
		std::vector<LandmarkObs> map_observations;

		for(auto &map_landmark : map_landmarks.landmark_list)
		{
			if(dist(particle.x, particle.y, map_landmark.x_f, map_landmark.y_f) < sensor_range)
			{
				predicted_distances.push_back(LandmarkObs{map_landmark.id_i, map_landmark.x_f, map_landmark.y_f});
			}
		}

		for(auto &observation : observations)
		{
			double x_map = particle.x + (cos(particle.theta) * observation.x - sin(particle.theta) * observation.y);
			double y_map = particle.y + (sin(particle.theta) * observation.x + cos(particle.theta) * observation.y);
			map_observations.push_back(LandmarkObs{0, x_map, y_map});
		}
		dataAssociation(predicted_distances, map_observations);
		flag = false;
		double sigmax = std_landmark[0];
		double sigmay = std_landmark[1];
		double sigmax_2 = sigmax * sigmax;
		double sigmay_2 = sigmay * sigmay;
		double two_sigmax_2 = 2 * sigmax_2;
		double two_sigmay_2 = 2 * sigmay_2;
		particle.weight = 1;
		double associated_id;
		for(size_t i = 0; i < map_observations.size() && i < predicted_distances.size() ; ++i)
		{
			associated_id = map_observations[i].id;
			double power = -1 * (pow(map_observations[i].x - predicted_distances[associated_id].x, 2) / two_sigmax_2
								 + pow(map_observations[i].y - predicted_distances[associated_id].y, 2) / two_sigmay_2);
			particle.weight *= (1 / (2 * M_PI * sigmax * sigmay)) * exp(power);
		}
		weights.push_back(particle.weight);
		if(particle.weight > max_weight)
		{
			max_weight = particle.weight;
		}
	}

}



void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	// http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	  default_random_engine generator;
	  uniform_int_distribution<int> distribution(0,num_particles);
	  uniform_real_distribution<double> uniform_distribution(0, 2 * max_weight);
	  double beta = 0;
	  long index = distribution(generator);
	  std::vector<Particle> resampled_particles;
	  for (long i = 0; i < num_particles; ++i) {
	    beta += uniform_distribution(generator);
	    while(weights[index] < beta)
	    {
	    	beta -= weights[index];
	    	index = (index + 1) % num_particles;
	    }
	    resampled_particles.push_back(particles[index]);
	  }
	  particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    // particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
